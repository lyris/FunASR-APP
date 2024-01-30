import os
import shutil
from contextlib import asynccontextmanager
from typing import Optional

import librosa
import torch as torch
import uvicorn
from fastapi import FastAPI, HTTPException, Form, UploadFile, File, status
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from videoclipper import VideoClipper

LOCAL_MODEL_DIR = 'd:\\temp\\modelscope\\hub\\damo\\'

path_asr = os.path.join(LOCAL_MODEL_DIR, 'speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
path_vad = os.path.join(LOCAL_MODEL_DIR, 'speech_fsmn_vad_zh-cn-16k-common-pytorch')
path_punc = os.path.join(LOCAL_MODEL_DIR, 'punc_ct-transformer_zh-cn-common-vocab272727-pytorch')
path_sd = os.path.join(LOCAL_MODEL_DIR, 'speech_campplus_speaker-diarization_common')

path_asr=path_asr if os.path.exists(path_asr)else "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
path_vad=path_vad if os.path.exists(path_vad)else "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
path_punc=path_punc if os.path.exists(path_punc)else "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
path_sd=path_sd if os.path.exists(path_sd)else "damo/speech_campplus_speaker-diarization_common"



inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model=path_asr,
    vad_model=path_vad,
    punc_model=path_punc,
    ncpu=16,
)
sd_pipeline = pipeline(
    task='speaker-diarization',
    model=path_sd,
    model_revision='v1.0.0'
)
audio_clipper = VideoClipper(inference_pipeline, sd_pipeline)


def audio_recog(audio_input, sd_switch):
    return audio_clipper.recog(audio_input, sd_switch)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


UPLOAD_DIR = "/tmp"
app = FastAPI(lifespan=lifespan)


@app.post('/v1/audio/transcriptions')
async def transcriptions(model: str = Form(...),
                         file: UploadFile = File(...),
                         response_format: Optional[str] = Form(None),
                         prompt: Optional[str] = Form(None),
                         sd_switch: Optional[str] = Form('yes')
                         ):
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bad Request, bad file"
        )
    if response_format is None:
        response_format = 'json'
    if response_format not in ['text',
                               'srt']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bad Request, bad response_format"
        )
    filename = file.filename
    fileobj = file.file
    upload_name = os.path.join(UPLOAD_DIR, filename)
    upload_file = open(upload_name, 'wb+')
    shutil.copyfileobj(fileobj, upload_file)
    upload_file.close()
    wav = librosa.load(upload_name, sr=16000)[0]
    res_text, res_srt, state = audio_recog(audio_input=(16000, wav), sd_switch=sd_switch)
    if response_format in ['text']:
        return res_text
    if response_format in ['srt']:
        return res_srt


if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=8001, timeout_keep_alive=900)
