# Vision Pro FFmpeg 8.1 Notes

This note captures practical command patterns validated for FFmpeg 8.1 workflows targeting Apple playback paths.

## Scope

- Focus: reliable 2D-to-3D and VR180 processing command paths using FFmpeg 8.1.
- Goal: avoid unsupported claims about native MV-HEVC and spatial metadata authoring in one FFmpeg CLI step.
- Output: reproducible prep assets plus explicit post-processing expectations.

## Capability Reality Check

Validated expectations for a practical FFmpeg 8.1 setup:

- Supported: v360 filter for projection conversion.
- Supported: stereo3d and framepack filters for stereo packaging transforms.
- Supported: hevc_videotoolbox encoder and hvc1 tag usage.
- Supported: hevc_metadata and dovi_rpu bitstream filters.
- Supported: ASS/SSA subtitle filter path only when libass is installed.
- Not expected in many 8.1 CLI builds: exposed MV-HEVC authoring controls in hevc_videotoolbox.
- Not expected in many 8.1 CLI builds: direct MOV option for Apple spatial video metadata injection.

## Build and Probe

Build:

- Use your FFmpegKit Apple build scripts as normal.
- Ensure libass is enabled if subtitle burn-in is required.

Probe commands:

- ffmpeg -hide_banner -h encoder=hevc_videotoolbox
- ffmpeg -hide_banner -h muxer=mov
- ffmpeg -hide_banner -filters | rg -i "v360|stereo3d|framepack|setparams"
- ffmpeg -hide_banner -bsfs | rg -i "hevc_metadata|dovi_rpu|hdr10"

## VR180 SBS to Vision Pro Prep

Use this for a robust preparation asset:

ffmpeg -i input_vr180_sbs.mp4 \
  -filter_complex "[0:v]crop=iw/2:ih:0:0,v360=input=hequirect:output=hequirect:h_fov=180:v_fov=180[left];[0:v]crop=iw/2:ih:iw/2:0,v360=input=hequirect:output=hequirect:h_fov=180:v_fov=180[right]" \
  -map "[left]" -map "[right]" -map 0:a? \
  -c:v hevc_videotoolbox -allow_sw 1 -profile:v main10 -tag:v hvc1 \
  -c:a copy -movflags +faststart \
  -metadata:s:v:0 title="Left Eye" -metadata:s:v:1 title="Right Eye" \
  output_visionpro_prep.mov

Notes:

- This creates a practical dual-eye HEVC prep container.
- Native Vision Pro spatial recognition generally still requires a post-encode metadata injector/tooling step.

## Color Metadata Patch Without Re-encode

When bitstream-compatible, patch HEVC VUI color signaling:

ffmpeg -i input.mov -map 0 -c copy \
  -bsf:v hevc_metadata=color_primaries=bt2020:transfer_characteristics=smpte2084:matrix_coefficients=bt2020nc:video_full_range_flag=0 \
  output_tagged.mov

## Output Verification

Recommended post-encode checks:

- ffprobe -hide_banner -v error -show_entries stream=index,codec_name,codec_tag_string,profile,width,height,pix_fmt,color_space,color_transfer,color_primaries:stream_tags=title -show_streams output_visionpro_prep.mov
- ffprobe -hide_banner -v error -select_streams v -show_frames -show_entries frame=side_data_list -of compact output_visionpro_prep.mov

## Subtitle and HDR Notes

- Subtitle rendering filter path needs libass. Container support alone is not enough for render-time subtitles filter usage.
- Dolby Vision RPU bitstream filter may be available even when external HDR10+ tooling path is absent.

## Practical Delivery Model

Use a two-step delivery model:

1. FFmpeg 8.1 for deterministic eye split/projection/encode and color tagging.
2. External Apple-compatible spatial metadata injection for final native spatial playback behavior.
