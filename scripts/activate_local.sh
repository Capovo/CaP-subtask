#!/usr/bin/env bash
# Source this file from Bash: source scripts/activate_local.sh
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Use: source /home/yilianli/project/CaP-subtask/scripts/activate_local.sh" >&2
  exit 1
fi

# Conda's CUDA activation hooks read optional variables without defaults.
# Restore the caller's nounset setting after those hooks finish.
_agenticwam_activate() {
  local _nounset_enabled=0 _activation_status=0
  case "$-" in
    *u*) _nounset_enabled=1; set +u ;;
  esac
  if source /opt/miniconda3/etc/profile.d/conda.sh && conda activate agenticwam; then
    :
  else
    _activation_status=$?
  fi
  if (( _nounset_enabled )); then
    set -u
  fi
  return "$_activation_status"
}
if ! _agenticwam_activate; then
  unset -f _agenticwam_activate
  return 1
fi
unset -f _agenticwam_activate

case ":${PYTHONPATH:-}:" in
  *:/home/yilianli/project/CaP-subtask/src:*) ;;
  *) export PYTHONPATH="/home/yilianli/project/CaP-subtask/src${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
export HF_HOME="${HF_HOME:-/home/yilianli/.cache/huggingface}"
export NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
for _agenticwam_host in localhost 127.0.0.1 ::1; do
  case ",${NO_PROXY}," in
    *,"$_agenticwam_host",*) ;;
    *) export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$_agenticwam_host" ;;
  esac
done
export no_proxy="$NO_PROXY"
unset _agenticwam_host
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export AGENTICWAM_DEPLOY_DTYPE="${AGENTICWAM_DEPLOY_DTYPE:-bf16}"

if [[ -x "$CONDA_PREFIX/bin/nvcc" ]]; then
  export CUDA_HOME="$CONDA_PREFIX"
fi

for _agenticwam_include in "$CONDA_PREFIX"/lib/python3.10/site-packages/nvidia/*/include; do
  [[ -d "$_agenticwam_include" ]] || continue
  case ":${CPATH:-}:" in
    *:"$_agenticwam_include":*) ;;
    *) export CPATH="$_agenticwam_include${CPATH:+:$CPATH}" ;;
  esac
done
unset _agenticwam_include

cd /home/yilianli/project/CaP-subtask || return
