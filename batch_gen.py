import sys
import os
from tqdm import tqdm
import glob
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

prompt_dir = 'asset'
prompt_files = glob.glob(os.path.join(prompt_dir, '*.wav'))
# prompt_speech_16k = load_wav('./asset/AI005_Jon.wav', 16000)
output_dir = 'generated_speeches'
os.makedirs(output_dir, exist_ok=True)

texts = [
    "Hey Memo. -- Take a picture. -- Stop recording.",
    "Hey Memo. -- Take a video. -- Stop recording.",
    "Hey Memo. -- Play. -- Pause.",
    "Hey Memo. -- Volume up. -- Next.",
    "Hey Memo. -- Volume down. -- Stop recording."
]

prompt_text = "."

# for i, text in enumerate(tqdm(texts)):
#     def text_generator():
#         yield text
#     for result in cosyvoice.inference_zero_shot(text_generator(), prompt_text, prompt_speech_16k, stream=False):
#         output_path = os.path.join(output_dir, f'speech_{i:03d}_{text[:20].replace(" ", "")}wav')
#         torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
#         print(f"已保存: {output_path}")

for prompt_file in tqdm(prompt_files):
    prompt_filename = os.path.splitext(os.path.basename(prompt_file))[0]
    prompt_speech_16k = load_wav(prompt_file, 16000)
    for i, text in enumerate(tqdm(texts, desc=f"生成 {prompt_filename} 的语音", leave=False)):
        def text_generator():
            yield text
        for result in cosyvoice.inference_zero_shot(text_generator(), prompt_text, prompt_speech_16k, stream=False):
            # 保存语音文件，文件名中包含提示音频的文件名和文本信息
            # output_path = os.path.join(output_dir, f'{prompt_filename}_{i:03d}_{text.replace(" ", "")}wav')
            output_path = os.path.join(output_dir, f'{prompt_filename}_{i:03d}_{prompt_index+1}.wav')
            torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
            print(f"已保存: {output_path}")

print(f"所有语音已生成并保存到 {output_dir} 目录")
