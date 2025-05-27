import traceback
import logging

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import torch
import torch_musa
from io import BytesIO
from datetime import datetime

from infer.lib.audio import load_audio, wav2
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import *


class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None

        self.config = config

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)
        
        # 清理 GPU 内存
        if hasattr(torch, "musa") and torch_musa.is_available():
            try:
                torch_musa.empty_cache()
                logger.info("Cleared MUSA GPU cache")
            except Exception as e:
                logger.warning(f"Failed to clear MUSA GPU cache: {str(e)}")

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33
            ),
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if (
                self.hubert_model is not None
            ):  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
                logger.info("Clean model cache")
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)  # ,cpt
                self.hubert_model = self.net_g = self.n_spk = self.hubert_model = (
                    self.tgt_sr
                ) = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ###楼下不这么折腾清理不干净
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                if self.version == "v1":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs256NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                elif self.version == "v2":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs768NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect0,
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": to_return_protect1,
                    "__type__": "update",
                },
                "",
                "",
            )
        person = f'{os.getenv("weight_root")}/{sid}'
        logger.info(f"Loading: {person}")

        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        logger.info("Select index: " + index["value"])

        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )

    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        if input_audio_path is None:
            return "You need to upload an audio", None
        try:
            # 清理 GPU 内存
            if hasattr(torch, "musa") and torch_musa.is_available():
                try:
                    torch_musa.empty_cache()
                    logger.info("Cleared MUSA GPU cache before processing")
                except Exception as e:
                    logger.warning(f"Failed to clear MUSA GPU cache: {str(e)}")

            f0_up_key = int(f0_up_key)
            # 确保 protect 有默认值
            if protect is None:
                protect = 0.33
            protect = float(protect)
            
            # 确保 resample_sr 有合理的默认值
            if resample_sr is None or resample_sr == 0:
                resample_sr = 44100  # 设置默认采样率为 44.1kHz
            resample_sr = int(resample_sr)
            
            logger.info(f"Processing audio: {input_audio_path}")
            logger.info(f"Using model: {sid}")
            logger.info(f"F0 method: {f0_method}")
            logger.info(f"Protect value: {protect}")
            logger.info(f"Resample rate: {resample_sr}")
            
            audio = load_audio(input_audio_path, 16000)
            if audio is None:
                logger.error("Failed to load audio")
                return "Failed to load audio", None
                
            # 检查并清理音频数据
            if not np.all(np.isfinite(audio)):
                logger.warning("Audio contains non-finite values, attempting to clean...")
                # 替换 NaN 和 Inf 值为 0
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                # 检查是否还有非有限值
                if not np.all(np.isfinite(audio)):
                    logger.error("Failed to clean audio data")
                    return "Audio data contains invalid values that cannot be cleaned", None
                logger.info("Audio data cleaned successfully")
            
            # 确保音频数据在合理范围内
            audio_max = np.abs(audio).max()
            if audio_max > 1:
                logger.info(f"Normalizing audio, max amplitude: {audio_max}")
                audio = audio / audio_max
            elif audio_max == 0:
                logger.error("Audio is silent")
                return "Audio is silent", None
                
            # 额外的数据验证
            if np.isnan(audio).any() or np.isinf(audio).any():
                logger.error("Audio still contains invalid values after cleaning")
                return "Audio data is invalid after cleaning", None
                
            # 检查音频长度
            if len(audio) < 16000:  # 小于1秒
                logger.error("Audio is too short")
                return "Audio is too short (less than 1 second)", None
                
            # 检查音频是否全为0
            if np.all(audio == 0):
                logger.error("Audio is completely silent")
                return "Audio is completely silent", None
                
            # 检查音频是否包含异常值
            if np.abs(audio).max() > 1e6:  # 异常大的值
                logger.error("Audio contains extreme values")
                return "Audio contains extreme values", None
                
            times = [0, 0, 0]

            if self.hubert_model is None:
                logger.info("Loading hubert model...")
                self.hubert_model = load_hubert(self.config)
                if self.hubert_model is None:
                    logger.error("Failed to load hubert model")
                    return "Failed to load hubert model", None

            if file_index:
                file_index = (
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""  # 防止小白写错，自动帮他替换掉

            logger.info("Starting pipeline processing...")
            try:
                audio_opt = self.pipeline.pipeline(
                    self.hubert_model,
                    self.net_g,
                    sid,
                    audio,
                    input_audio_path,
                    times,
                    f0_up_key,
                    f0_method,
                    file_index,
                    index_rate,
                    self.if_f0,
                    filter_radius,
                    self.tgt_sr,
                    resample_sr,
                    rms_mix_rate,
                    self.version,
                    protect,
                    f0_file,
                )
            except RuntimeError as e:
                if "MUSA error" in str(e):
                    logger.error(f"MUSA GPU error occurred: {str(e)}, trying to recover...")
                    # 尝试清理 GPU 内存并重新加载模型
                    if hasattr(torch, "musa") and torch_musa.is_available():
                        try:
                            torch_musa.empty_cache()
                            logger.info("Cleared GPU cache, reloading model...")
                            # 重新加载模型
                            if isinstance(sid, int):
                                sid = str(sid)
                            self.get_vc(sid)
                            logger.info("Model reloaded, retrying inference...")
                            # 重试推理
                            audio_opt = self.pipeline.pipeline(
                                self.hubert_model,
                                self.net_g,
                                sid,
                                audio,
                                input_audio_path,
                                times,
                                f0_up_key,
                                f0_method,
                                file_index,
                                index_rate,
                                self.if_f0,
                                filter_radius,
                                self.tgt_sr,
                                resample_sr,
                                rms_mix_rate,
                                self.version,
                                protect,
                                f0_file,
                            )
                        except Exception as retry_e:
                            error_msg = f"MUSA GPU error: {str(e)}. Recovery failed: {str(retry_e)}"
                            logger.error(error_msg)
                            return error_msg, None
                else:
                    raise e
            
            if audio_opt is None:
                logger.error("Pipeline processing returned None")
                return "Pipeline processing failed", None
                
            logger.info("Pipeline processing completed successfully")
            logger.info(f"Processing time: {times[0]:.2f}s")
            logger.info(f"Output sample rate: {resample_sr}")
            
            # 验证输出音频
            if audio_opt is not None:
                if not np.all(np.isfinite(audio_opt)):
                    logger.warning("Output audio contains non-finite values, cleaning...")
                    audio_opt = np.nan_to_num(audio_opt, nan=0.0, posinf=0.0, neginf=0.0)
                    if not np.all(np.isfinite(audio_opt)):
                        logger.error("Failed to clean output audio")
                        return "Failed to clean output audio", None
                
                # 确保输出音频在合理范围内
                audio_max = np.abs(audio_opt).max()
                if audio_max > 1:
                    logger.info(f"Normalizing output audio, max amplitude: {audio_max}")
                    audio_opt = audio_opt / audio_max
                
                # 自动保存处理后的音频
                try:
                    # 获取输入音频的目录和文件名
                    input_dir = os.path.dirname(input_audio_path)
                    input_filename = os.path.basename(input_audio_path)
                    input_name, input_ext = os.path.splitext(input_filename)
                    
                    # 生成带时间戳的输出文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"{input_name}_converted_{timestamp}{input_ext}"
                    output_path = os.path.join(input_dir, output_filename)
                    
                    # 保存音频文件
                    logger.info(f"Saving converted audio to: {output_path}")
                    sf.write(output_path, audio_opt, resample_sr)
                    logger.info("Audio saved successfully")
                    
                    # 返回成功信息和保存路径
                    return (
                        f"Success, saved to: {output_path}",
                        (resample_sr, audio_opt),
                    )
                except Exception as save_error:
                    logger.error(f"Error saving audio: {str(save_error)}")
                    # 即使保存失败也返回音频数据，这样 Gradio 界面仍然可以播放
                    return (
                        f"Success but save failed: {str(save_error)}",
                        (resample_sr, audio_opt),
                    )
            
            return (
                "Success",
                (resample_sr, audio_opt),
            )
            
        except Exception as e:
            logger.error(f"Error in vc_single: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error: {str(e)}", None

    def vc_multi(
        self,
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
    ):
        try:
            dir_path = (
                dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )  # 防止小白拷路径头尾带了空格和"和回车
            opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            os.makedirs(opt_root, exist_ok=True)
            try:
                if dir_path != "":
                    paths = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                else:
                    paths = [path.name for path in paths]
            except:
                traceback.print_exc()
                paths = [path.name for path in paths]
            infos = []
            for path in paths:
                info, opt = self.vc_single(
                    sid,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    file_index,
                    file_index2,
                    # file_big_npy,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                )
                if "Success" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        if format1 in ["wav", "flac"]:
                            sf.write(
                                "%s/%s.%s"
                                % (opt_root, os.path.basename(path), format1),
                                audio_opt,
                                tgt_sr,
                            )
                        else:
                            path = "%s/%s.%s" % (
                                opt_root,
                                os.path.basename(path),
                                format1,
                            )
                            with BytesIO() as wavf:
                                sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                wavf.seek(0, 0)
                                with open(path, "wb") as outf:
                                    wav2(wavf, outf, format1)
                    except:
                        info += traceback.format_exc()
                infos.append("%s->%s" % (os.path.basename(path), info))
                yield "\n".join(infos)
            yield "\n".join(infos)
        except:
            yield traceback.format_exc()
