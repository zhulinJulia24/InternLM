import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

prompts = ['你好', "what's your name"]


def assert_model(response):
    assert len(response) != 0
    assert 'UNUSED_TOKEN' not in response
    assert 'Mynameis' not in response
    assert 'Iama' not in response


class TestChat:

    @pytest.mark.parametrize('model_name', ['internlm/Intern-S1'])
    @pytest.mark.parametrize('enable_thinking', [True, False])
    @pytest.mark.gpu_num_8
    @pytest.mark.interns1
    def test_demo_default_gpu8(self, model_name, enable_thinking, worker_id):
        test_s1_chat_demo(model_name, enable_thinking)

    @pytest.mark.parametrize('model_name', ['internlm/Intern-S1-mini'])
    @pytest.mark.parametrize('enable_thinking', [True, False])
    @pytest.mark.gpu_num_1
    @pytest.mark.interns1
    def test_demo_default_gpu1(self, model_name, enable_thinking, worker_id):
        test_s1_chat_demo(model_name, enable_thinking)


def test_s1_chat_demo(model_name, enable_thinking):
    processor = AutoProcessor.from_pretrained(model_name,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map='auto',
                                                 torch_dtype='auto',
                                                 trust_remote_code=True)
    prompts = [
        'tell me about an interesting physical phenomenon.', '请给我讲一个有趣的物理现象'
    ]
    for prompt in prompts:
        messages = [{
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': prompt
            }]
        }]

        inputs = processor.apply_chat_template(messages,
                                               add_generation_prompt=True,
                                               tokenize=True,
                                               return_dict=True,
                                               return_tensors='pt').to(
                                                   model.device,
                                                   dtype=torch.bfloat16)

        generate_ids = model.generate(**inputs, max_new_tokens=32768)
        decoded_output = processor.decode(
            generate_ids[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True)
        assert 'physical phenomenon' in decoded_output.lower(), decoded_output
        assert_model(decoded_output)


def get_cuda_prefix_by_workerid(worker_id, tp_num: int = 1):
    cuda_id = get_cuda_id_by_workerid(worker_id, tp_num)
    if cuda_id is None or 'gw' not in worker_id:
        return None
    else:
        device_type = os.environ.get('DEVICE', 'cuda')
        if device_type == 'ascend':
            return 'ASCEND_RT_VISIBLE_DEVICES=' + cuda_id
        else:
            return 'CUDA_VISIBLE_DEVICES=' + cuda_id


def get_cuda_id_by_workerid(worker_id, tp_num: int = 1):
    if worker_id is None or 'gw' not in worker_id:
        return None
    else:
        if tp_num == 1:
            return worker_id.replace('gw', '')
        elif tp_num == 2:
            cuda_num = int(worker_id.replace('gw', '')) * 2
            return ','.join([str(cuda_num), str(cuda_num + 1)])
        elif tp_num == 4:
            cuda_num = int(worker_id.replace('gw', '')) * 4
            return ','.join([
                str(cuda_num),
                str(cuda_num + 1),
                str(cuda_num + 2),
                str(cuda_num + 3)
            ])


def set_device_env_variable(worker_id, tp_num: int = 1):
    cuda_id = get_cuda_id_by_workerid(worker_id, tp_num)
    if cuda_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_id


def unset_device_env_variable():
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
