###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
#
#  MuseTalk 数字人 — 迁移自 musereal.py + museasr.py
#

import math
import torch
import numpy as np

import subprocess
import os
import time
import torch.nn.functional as F
import cv2
import glob
import pickle
import copy

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from avatars.musetalk.utils.utils import get_file_type,get_video_fps,datagen
from avatars.musetalk.myutil import get_image_blending
from avatars.musetalk.utils.utils import load_all_model
from avatars.musetalk.whisper.audio2feature import Audio2Feature

from avatars.audio_features.whisper import WhisperASR
import asyncio
from av import AudioFrame, VideoFrame
from avatars.base_avatar import BaseAvatar, AudioFrameData

from tqdm import tqdm
from utils.logger import logger
from utils.image import read_imgs, mirror_index
from utils.device import initialize_device
from utils.trace import mark
from registry import register

device = initialize_device()
logger.info('Using {} for inference.'.format(device))

def load_model():
    # load model weights
    vae, unet, pe = load_all_model()
    #device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"))
    timesteps = torch.tensor([0], device=device)
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    # Initialize audio processor and Whisper model
    audio_processor = Audio2Feature(model_path="./models/whisper")
    return vae, unet, pe, timesteps, audio_processor

def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    latents_out_path= f"{avatar_path}/latents.pt"
    video_out_path = f"{avatar_path}/vid_output/"
    mask_out_path =f"{avatar_path}/mask"
    mask_coords_path =f"{avatar_path}/mask_coords.pkl"
    avatar_info_path = f"{avatar_path}/avator_info.json"

    input_latent_list_cycle = torch.load(latents_out_path, map_location=device)
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    frame_list_cycle = None
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    with open(mask_coords_path, 'rb') as f:
        mask_coords_list_cycle = pickle.load(f)
    input_mask_list = glob.glob(os.path.join(mask_out_path, '*.[jpJP][pnPN]*[gG]'))
    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    mask_list_cycle = read_imgs(input_mask_list)
    return frame_list_cycle,mask_list_cycle,coord_list_cycle,mask_coords_list_cycle,input_latent_list_cycle

@torch.no_grad()
def warm_up(batch_size,model):
    # 预热函数
    print('warmup model...')
    vae, unet, pe, timesteps, audio_processor = model
    whisper_batch = np.ones((batch_size, 50, 384), dtype=np.uint8)
    latent_batch = torch.ones(batch_size, 8, 32, 32).to(unet.device)

    audio_feature_batch = torch.from_numpy(whisper_batch)
    audio_feature_batch = audio_feature_batch.to(device=unet.device, dtype=unet.model.dtype)
    audio_feature_batch = pe(audio_feature_batch)
    latent_batch = latent_batch.to(dtype=unet.model.dtype)
    pred_latents = unet.model(latent_batch,
                              timesteps,
                              encoder_hidden_states=audio_feature_batch).sample
    vae.decode_latents(pred_latents)    

@register("avatar", "musetalk")
class MuseReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        #self.fps = opt.fps # 20 ms per frame

        # self.batch_size = opt.batch_size
        # self.idx = 0
        # self.res_frame_queue = mp.Queue(self.batch_size*2)

        self.vae, self.unet, self.pe, self.timesteps, self.audio_processor = model

        self.frame_list_cycle,self.mask_list_cycle,self.coord_list_cycle,self.mask_coords_list_cycle, self.input_latent_list_cycle = avatar

        self.asr = WhisperASR(opt,self,self.audio_processor)
        self.asr.warm_up()
    

    def inference_batch(self, index, audiofeat_batch):
        # 这里的 index 是针对当前 avatar 的索引
        # 返回一个 batch 的推理结果，batch 大小由 audiofeat_batch 决定
        length = len(self.input_latent_list_cycle)
        whisper_batch = np.stack(audiofeat_batch)
        latent_batch = []
        batch_size = len(audiofeat_batch)
        for i in range(batch_size):
            idx = mirror_index(length, index + i)
            latent = self.input_latent_list_cycle[idx]
            latent_batch.append(latent)
        latent_batch = torch.cat(latent_batch, dim=0)
        
        audio_feature_batch = torch.from_numpy(whisper_batch)
        audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                                                        dtype=self.unet.model.dtype)
        audio_feature_batch = self.pe(audio_feature_batch)
        latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

        pred_latents = self.unet.model(latent_batch, 
                                    self.timesteps, 
                                    encoder_hidden_states=audio_feature_batch).sample
        pred = self.vae.decode_latents(pred_latents)
        return pred

    def inference_batch_tensor(self, index, audio_feature_batch):
        length = len(self.input_latent_list_cycle)
        batch_size = audio_feature_batch.shape[0]
        latent_batch = []
        for i in range(batch_size):
            idx = mirror_index(length, index + i)
            latent_batch.append(self.input_latent_list_cycle[idx])
        latent_batch = torch.cat(latent_batch, dim=0)

        audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                                                     dtype=self.unet.model.dtype)
        audio_feature_batch = self.pe(audio_feature_batch)
        latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

        pred_latents = self.unet.model(latent_batch,
                                    self.timesteps,
                                    encoder_hidden_states=audio_feature_batch).sample
        pred = self.vae.decode_latents(pred_latents)
        return pred

    def paste_back_frame(self,pred_frame,idx:int):
        bbox = self.coord_list_cycle[idx]
        ori_frame = copy.deepcopy(self.frame_list_cycle[idx])
        x1, y1, x2, y2 = bbox

        res_frame = cv2.resize(pred_frame.astype(np.uint8),(x2-x1,y2-y1))
        mask = self.mask_list_cycle[idx]
        mask_crop_box = self.mask_coords_list_cycle[idx]

        combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)
        return combine_frame


@register("avatar", "musetalk_fused")
class MuseRealFused(MuseReal):
    def __init__(self, opt, model, avatar):
        super().__init__(opt, model, avatar)
        self._drain_asr_output_queue()

    def _drain_asr_output_queue(self):
        while True:
            try:
                self.asr.output_queue.get_nowait()
            except queue.Empty:
                return

    def _feature2chunks_tensor(self, feature_array, batch_size, audio_feat_win=[0, 5],
                               start=0, feature_idx_multiplier=2):
        feature_chunks = []
        length = feature_array.shape[0]
        for i in range(batch_size):
            center_idx = int((i + start) * feature_idx_multiplier)
            left = int(center_idx - audio_feat_win[0] * feature_idx_multiplier)
            right = int(center_idx + audio_feat_win[1] * feature_idx_multiplier)
            selected_idx = [min(max(idx, 0), length - 1) for idx in range(left, right)]
            selected_idx = torch.tensor(selected_idx, device=feature_array.device, dtype=torch.long)
            selected_feature = torch.index_select(feature_array, 0, selected_idx)
            feature_chunks.append(selected_feature.reshape(-1, 384))
        return torch.stack(feature_chunks, dim=0)

    def _fused_step(self, index, batch_size, last_speaking):
        start_time = time.perf_counter()
        audio_frames: list[AudioFrameData] = []
        is_all_silence = True

        for _ in range(batch_size * 2):
            audio_frame = self.asr.get_audio_frame()
            if audio_frame.type == 0:
                is_all_silence = False
            self.asr.frames.append(audio_frame.data)
            audio_frames.append(audio_frame)

        if len(self.asr.frames) <= self.asr.stride_left_size + self.asr.stride_right_size:
            return index, last_speaking, 0, 0

        current_batch_size = batch_size
        first_trace = next(
            (frame.userdata for frame in audio_frames if frame.type == 0 and frame.userdata.get("_trace_start")),
            None,
        )
        if first_trace:
            mark(
                first_trace,
                "avatar.infer.first_audio_batch",
                detail=f"batch_audio_frames={len(audio_frames)} res_qsize={self.res_frame_queue.qsize()}",
                once_key="avatar_first_audio_batch",
            )

        current_speaking = not is_all_silence
        length = self.get_avatar_length()

        if is_all_silence:
            for i in range(current_batch_size):
                idx = mirror_index(length, index)
                if first_trace and i == 0:
                    mark(
                        first_trace,
                        "avatar.infer.first_silence_frame_enqueue",
                        detail=f"res_qsize={self.res_frame_queue.qsize()}",
                        once_key="avatar_first_frame_enqueue",
                    )
                self.res_frame_queue.put((None, audio_frames[i * 2:i * 2 + 2], idx))
                index = index + 1
        else:
            if current_speaking and not last_speaking and self.custom_index.get(1) is not None:
                index = 0

            t = time.perf_counter()
            inputs = np.concatenate(self.asr.frames)
            whisper_feature = self.audio_processor.audio2feat_tensor(inputs)
            audio_feature_batch = self._feature2chunks_tensor(
                feature_array=whisper_feature,
                batch_size=current_batch_size,
                audio_feat_win=[0, 5],
                start=self.asr.stride_left_size / 2,
                feature_idx_multiplier=2,
            )
            if first_trace:
                mark(
                    first_trace,
                    "asr.whisper.first_feat_enqueue",
                    detail=(
                        f"batch={audio_feature_batch.shape[0]} fused=True "
                        f"cost_ms={(time.perf_counter() - t) * 1000:.1f}"
                    ),
                    once_key="asr_first_feat_enqueue",
                )

            if first_trace:
                mark(
                    first_trace,
                    "avatar.infer.first_model_start",
                    detail=f"batch_size={current_batch_size} fused=True",
                    once_key="avatar_first_model_start",
                )

            pred = self.inference_batch_tensor(index, audio_feature_batch)
            infer_cost = time.perf_counter() - t

            if first_trace:
                mark(
                    first_trace,
                    "avatar.infer.first_model_done",
                    detail=f"cost_ms={infer_cost * 1000:.1f} fused=True",
                    once_key="avatar_first_model_done",
                )

            for i, res_frame in enumerate(pred):
                if first_trace and i == 0:
                    mark(
                        first_trace,
                        "avatar.infer.first_frame_enqueue",
                        detail=f"res_qsize={self.res_frame_queue.qsize()} fused=True",
                        once_key="avatar_first_frame_enqueue",
                    )
                self.res_frame_queue.put((res_frame, audio_frames[i * 2:i * 2 + 2], mirror_index(length, index)))
                index = index + 1

        self.asr.frames = self.asr.frames[-(self.asr.stride_left_size + self.asr.stride_right_size):]

        if current_speaking != last_speaking:
            logger.info(f"fused inference 状态切换：{'说话' if last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
            last_speaking = current_speaking

        return index, last_speaking, current_batch_size, time.perf_counter() - start_time

    def render(self, quit_event):
        self.quit_event = quit_event

        self.init_customindex()
        self.tts.render(quit_event)

        process_quit_event = Event()
        process_thread = Thread(target=self.process_frames, args=(process_quit_event,))
        process_thread.start()

        index = 0
        count = 0
        counttime = 0
        last_speaking = False
        logger.info('start musetalk fused render')

        while not quit_event.is_set():
            if self.low_latency and self._low_latency_pending_first_frame:
                current_batch_size = min(self.low_latency_batch_size, self.batch_size)
            else:
                current_batch_size = self.batch_size

            index, last_speaking, processed, step_cost = self._fused_step(
                index,
                current_batch_size,
                last_speaking,
            )
            if processed:
                count += processed
                counttime += step_cost
                if count >= 100:
                    logger.info(f"------actual avg fused fps:{count/counttime:.4f}")
                    count = 0
                    counttime = 0

            buffer_size = self.output.get_buffer_size() if hasattr(self.output, 'get_buffer_size') else 0
            if buffer_size >= 5:
                logger.debug('sleep qsize=%d', buffer_size)
                sleep_time = 0.04 * buffer_size * 0.8
                if self.low_latency and self._low_latency_pending_first_frame:
                    if getattr(self.asr, "queue", None) is not None and self.asr.queue.qsize() > 0:
                        sleep_time = 0
                    else:
                        sleep_time = min(sleep_time, 0.02)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        logger.info('musetalk fused render thread stop')
        process_quit_event.set()
        process_thread.join()
            
