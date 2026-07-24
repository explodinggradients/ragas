import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_llms_base_imports_without_vertexai_provider():
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    code = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("langchain_google_vertexai"):
                raise ImportError(name)
            if name == "langchain_community.chat_models.vertexai":
                raise ImportError(name)
            if name == "langchain_community.llms" and "VertexAI" in fromlist:
                raise ImportError(name)
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = fake_import

        import ragas.llms.base as base

        assert base.ChatVertexAI is None
        assert base.VertexAI is None
        assert None not in base.MULTIPLE_COMPLETION_SUPPORTED
        """
    )

    subprocess.run([sys.executable, "-c", code], env=env, check=True)
