import os
from utils import (
    get_cuda_id_by_workerid,
    get_cuda_prefix_by_workerid,
    set_device_env_variable,
    unset_device_env_variable,
)

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

prompts = ['你好', "what's your name"]

@pytest.fixture(scope='function', autouse=True)
def device_setup(request, worker_id):
    if request.node.get_closest_marker('gpu_num_8'):
        tp_num = 8
    elif request.node.get_closest_marker('gpu_num_4'):
        tp_num = 4
    elif request.node.get_closest_marker('gpu_num_2'):
        tp_num = 2
    elif request.node.get_closest_marker('gpu_num_1'):
        tp_num = 1
    else:
        tp_num = 1
    
    set_device_env_variable(worker_id, tp_num)
    yield
    unset_device_env_variable()


@pytest.mark.parametrize('model_name', ['internlm/Intern-S1'])
@pytest.mark.parametrize('enable_thinking', [True, False])
@pytest.mark.gpu_num_8
@pytest.mark.interns1
def test_demo_default_gpu8(model_name, enable_thinking, device_setup):
    test_s1_chat_text_demo(model_name, enable_thinking)
    test_s1_chat_image_demo(model_name, enable_thinking)
    test_s1_chat_video_demo(model_name, enable_thinking)

@pytest.mark.parametrize('model_name', ['internlm/Intern-S1-mini'])
@pytest.mark.parametrize('enable_thinking', [True, False])
@pytest.mark.gpu_num_1
@pytest.mark.interns1
def test_demo_default_gpu1(model_name, enable_thinking, device_setup):
    test_s1_chat_text_demo(model_name, enable_thinking)
    test_s1_chat_image_demo(model_name, enable_thinking)
    test_s1_chat_video_demo(model_name, enable_thinking)

def test_s1_chat_text_demo(model_name, enable_thinking):
    return

def test_s1_chat_image_demo(model_name, enable_thinking):
    return

def test_s1_chat_video_demo(model_name, enable_thinking):
    return


def test_s1_functioncall_demo(model_name, enable_thinking):
    return

