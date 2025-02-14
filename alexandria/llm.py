from langchain.llms.base import LLM
import subprocess

class DeepseekLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "deepseek_r1"

    def _call(self, prompt: str, stop=None) -> str:
        print(f"[DeepseekLLM] Received prompt: {prompt}")  # Debug logging
        try:
            result = subprocess.run(
                ["/usr/bin/docker", "exec", "ollama", "run", "deepseek-r1", prompt],
                capture_output=True,
                text=True,
                check=True
)

            output = result.stdout.strip()
            print(f"[DeepseekLLM] Output: {output}")  # Debug logging
            return output
        except Exception as e:
            err_msg = f"Error calling Deepseek R1: {str(e)}"
            print(f"[DeepseekLLM] {err_msg}")
            return err_msg