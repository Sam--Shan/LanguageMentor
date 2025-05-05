import unittest
import json
import os
from unittest.mock import MagicMock, patch, mock_open
from langchain_core.messages import HumanMessage

# 被测试的模块需要导入
from agents.agent_base import AgentBase  # 根据实际路径调整

class TestConcreteAgent(AgentBase):
    """用于测试的具体代理类"""
    @abstractmethod
    def abstract_method(self):
        pass  # 仅用于实例化测试

class TestAgentBase(unittest.TestCase):
    def setUp(self):
        # Mock 文件内容
        self.mock_prompt = "Test system prompt"
        self.mock_intro = [{"role": "user", "content": "Hello"}]
        
        # 创建临时测试文件
        self.prompt_file = "test_prompt.txt"
        with open(self.prompt_file, "w") as f:
            f.write(self.mock_prompt)
            
        self.intro_file = "test_intro.json"
        with open(self.intro_file, "w") as f:
            json.dump(self.mock_intro, f)

        # 模拟 ChatOllama 和 RunnableWithMessageHistory
        self.mock_chatbot = MagicMock()
        self.mock_chatbot_with_history = MagicMock()
        self.mock_response = MagicMock()
        self.mock_response.content = "Mocked response"

    def tearDown(self):
        # 清理临时文件
        if os.path.exists(self.prompt_file):
            os.remove(self.prompt_file)
        if os.path.exists(self.intro_file):
            os.remove(self.intro_file)

    @patch.object(AgentBase, 'create_chatbot')
    def test_initialization(self, mock_create_chatbot):
        """测试类初始化参数"""
        agent = TestConcreteAgent(
            name="test_agent",
            prompt_file=self.prompt_file,
            intro_file=self.intro_file,
            session_id="test_session"
        )
        
        self.assertEqual(agent.name, "test_agent")
        self.assertEqual(agent.prompt, self.mock_prompt)
        self.assertEqual(agent.intro_messages, self.mock_intro)
        self.assertEqual(agent.session_id, "test_session")
        mock_create_chatbot.assert_called_once()

    def test_default_session_id(self):
        """测试默认 session_id"""
        agent = TestConcreteAgent(
            name="test_agent",
            prompt_file=self.prompt_file
        )
        self.assertEqual(agent.session_id, "test_agent")

    def test_load_prompt_success(self):
        """测试正常加载提示文件"""
        agent = TestConcreteAgent(
            name="test_agent",
            prompt_file=self.prompt_file
        )
        self.assertEqual(agent.prompt, self.mock_prompt)

    def test_load_prompt_file_not_found(self):
        """测试提示文件不存在异常"""
        with self.assertRaises(FileNotFoundError):
            TestConcreteAgent(
                name="test_agent",
                prompt_file="non_existent.txt"
            )

    def test_load_intro_success(self):
        """测试正常加载介绍文件"""
        agent = TestConcreteAgent(
            name="test_agent",
            prompt_file=self.prompt_file,
            intro_file=self.intro_file
        )
        self.assertEqual(agent.intro_messages, self.mock_intro)

    def test_load_intro_invalid_json(self):
        """测试无效JSON异常"""
        invalid_file = "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("{invalid json}")
            
        with self.assertRaises(ValueError):
            TestConcreteAgent(
                name="test_agent",
                prompt_file=self.prompt_file,
                intro_file=invalid_file
            )
        os.remove(invalid_file)

    @patch('langchain_core.prompts.ChatPromptTemplate.from_messages')
    @patch('langchain_ollama.chat_models.ChatOllama')
    def test_create_chatbot(self, mock_chat_ollama, mock_prompt_template):
        """测试聊天机器人创建流程"""
        mock_prompt = MagicMock()
        mock_prompt_template.return_value = mock_prompt
        
        agent = TestConcreteAgent(
            name="test_agent",
            prompt_file=self.prompt_file
        )
        
        mock_prompt_template.assert_called_once_with([
            ("system", self.mock_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        mock_chat_ollama.assert_called_once_with(
            model="llama3.1:8b-instruct-q8_0",
            max_tokens=8192,
            temperature=0.8
        )

    @patch('utils.logger.LOG.debug')
    def test_chat_with_history(self, mock_log):
        """测试带历史记录的聊天流程"""
        # 配置 mock 对象
        mock_invoke = MagicMock(return_value=self.mock_response)
        agent = TestConcreteAgent(
            name="test_agent",
            prompt_file=self.prompt_file
        )
        agent.chatbot_with_history = MagicMock(invoke=mock_invoke)

        # 测试调用
        result = agent.chat_with_history("test input", session_id="custom_session")
        
        # 验证调用参数
        mock_invoke.assert_called_once_with(
            [HumanMessage(content="test input")],
            {"configurable": {"session_id": "custom_session"}}
        )
        self.assertEqual(result, "Mocked response")
        mock_log.assert_called_once_with(
            f"[ChatBot][test_agent] Mocked response"
        )

    def test_chat_default_session_id(self):
        """测试默认session_id的使用"""
        agent = TestConcreteAgent(
            name="test_agent",
            prompt_file=self.prompt_file
        )
        agent.chatbot_with_history = MagicMock()
        agent.chatbot_with_history.invoke.return_value = self.mock_response

        agent.chat_with_history("test input")
        
        agent.chatbot_with_history.invoke.assert_called_once_with(
            [HumanMessage(content="test input")],
            {"configurable": {"session_id": "test_agent"}}
        )

if __name__ == '__main__':
    unittest.main()