import pytest

prompts = ['你好', "what's your name"]


@pytest.mark.parametrize('model_name', ['internlm/Intern-S1'])
@pytest.mark.parametrize('enable_thinking', [True, False])
@pytest.mark.gpu_num_8
@pytest.mark.interns1
def test_demo_default_gpu8(model_name, enable_thinking):
    test_s1_chat_text_demo(model_name, enable_thinking)
    test_s1_chat_image_demo(model_name, enable_thinking)
    test_s1_chat_video_demo(model_name, enable_thinking)


@pytest.mark.parametrize('model_name', ['internlm/Intern-S1-mini'])
@pytest.mark.parametrize('enable_thinking', [True, False])
@pytest.mark.gpu_num_1
@pytest.mark.interns1
def test_demo_default_gpu1(model_name, enable_thinking):
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
