#!/usr/bin/env bash
set -euo pipefail

FF="${1:-ffmpeg}"

echo "== decoder: hevc mv-hevc hints =="
"$FF" -hide_banner -h decoder=hevc 2>/dev/null | rg -i "view_ids|MV-HEVC" || true

echo "== mux/demux: iamf =="
"$FF" -hide_banner -muxers 2>/dev/null | rg -i iamf || true
"$FF" -hide_banner -demuxers 2>/dev/null | rg -i iamf || true

echo "== filters: projection/stereo/quality/subtitles =="
"$FF" -hide_banner -filters 2>/dev/null | rg -i -e "(^| )v360( |$)|(^| )stereo3d( |$)|(^| )framepack( |$)|(^| )zscale( |$)|(^| )vmaf( |$)|(^| )ass( |$)|(^| )subtitles( |$)" || true

echo "== bsf: metadata paths =="
"$FF" -hide_banner -bsfs 2>/dev/null | rg -i -e "hevc_metadata|dovi_rpu|hdr10" || true

echo "== mov muxer options: color/spatial hints =="
"$FF" -hide_banner -h muxer=mov 2>/dev/null | rg -i "write_colr|prefer_icc|master_display|content_light|spatial|stereo|sv3d" || true

echo "== encoder: hevc_videotoolbox =="
"$FF" -hide_banner -h encoder=hevc_videotoolbox 2>/dev/null | rg -i "profile|view|multiview|stereo|hdr|dolby" || true
