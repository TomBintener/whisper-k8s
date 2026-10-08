"""
Mock definitions for external dependencies (kubernetes, fastapi, httpx, pydantic)
when running tests in environments where these packages are not installed globally.
"""

import sys
from types import ModuleType
from typing import Any, Dict, Optional


class SimpleRecord:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"


# ---------------------------------------------------------------------------
# Mock kubernetes
# ---------------------------------------------------------------------------
if "kubernetes" not in sys.modules:
    try:
        import kubernetes  # type: ignore
    except ImportError:
        k8s_mod = ModuleType("kubernetes")
        client_mod = ModuleType("kubernetes.client")
        config_mod = ModuleType("kubernetes.config")

        class ApiException(Exception):
            def __init__(self, status=500, reason="API Error"):
                super().__init__(reason)
                self.status = status
                self.reason = reason

        class V1EnvVar(SimpleRecord): pass
        class V1EnvVarSource(SimpleRecord): pass
        class V1ObjectFieldSelector(SimpleRecord): pass
        class V1ResourceRequirements(SimpleRecord): pass
        class V1Volume(SimpleRecord): pass
        class V1VolumeMount(SimpleRecord): pass
        class V1PersistentVolumeClaimVolumeSource(SimpleRecord): pass
        class V1SecretVolumeSource(SimpleRecord): pass
        class V1ProjectedVolumeSource(SimpleRecord): pass
        class V1ConfigMapProjection(SimpleRecord): pass
        class V1VolumeProjection(SimpleRecord): pass
        class V1Container(SimpleRecord): pass
        class V1ObjectMeta(SimpleRecord): pass
        class V1PodSpec(SimpleRecord): pass
        class V1PodTemplateSpec(SimpleRecord): pass
        class V1JobSpec(SimpleRecord): pass
        class V1Job(SimpleRecord): pass
        class BatchV1Api(SimpleRecord):
            def list_namespaced_job(self, *args, **kwargs): return SimpleRecord(items=[])
            def create_namespaced_job(self, *args, **kwargs): return None
            def delete_namespaced_job(self, *args, **kwargs): return None
            def read_namespaced_job(self, *args, **kwargs): return None

        class CoreV1Api(SimpleRecord):
            def read_namespaced_config_map(self, *args, **kwargs): return None

        client_mod.ApiException = ApiException
        client_mod.V1EnvVar = V1EnvVar
        client_mod.V1EnvVarSource = V1EnvVarSource
        client_mod.V1ObjectFieldSelector = V1ObjectFieldSelector
        client_mod.V1ResourceRequirements = V1ResourceRequirements
        client_mod.V1Volume = V1Volume
        client_mod.V1VolumeMount = V1VolumeMount
        client_mod.V1PersistentVolumeClaimVolumeSource = V1PersistentVolumeClaimVolumeSource
        client_mod.V1SecretVolumeSource = V1SecretVolumeSource
        client_mod.V1ProjectedVolumeSource = V1ProjectedVolumeSource
        client_mod.V1ConfigMapProjection = V1ConfigMapProjection
        client_mod.V1VolumeProjection = V1VolumeProjection
        client_mod.V1Container = V1Container
        client_mod.V1ObjectMeta = V1ObjectMeta
        client_mod.V1PodSpec = V1PodSpec
        client_mod.V1PodTemplateSpec = V1PodTemplateSpec
        client_mod.V1JobSpec = V1JobSpec
        client_mod.V1Job = V1Job
        client_mod.BatchV1Api = BatchV1Api
        client_mod.CoreV1Api = CoreV1Api

        config_mod.load_incluster_config = lambda: None
        config_mod.load_kube_config = lambda: None

        k8s_mod.client = client_mod
        k8s_mod.config = config_mod

        sys.modules["kubernetes"] = k8s_mod
        sys.modules["kubernetes.client"] = client_mod
        sys.modules["kubernetes.config"] = config_mod


# ---------------------------------------------------------------------------
# Mock pydantic
# ---------------------------------------------------------------------------
if "pydantic" not in sys.modules:
    try:
        import pydantic  # type: ignore
    except ImportError:
        pydantic_mod = ModuleType("pydantic")

        def model_validator(*args, **kwargs):
            def decorator(fn):
                fn._is_model_validator = True
                return fn
            return decorator

        class BaseModel:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
                # Call any methods decorated with model_validator
                for attr_name in dir(self):
                    attr = getattr(self, attr_name)
                    if callable(attr) and getattr(attr, "_is_model_validator", False):
                        attr()

        pydantic_mod.BaseModel = BaseModel
        pydantic_mod.model_validator = model_validator
        sys.modules["pydantic"] = pydantic_mod


# ---------------------------------------------------------------------------
# Mock fastapi & httpx
# ---------------------------------------------------------------------------
if "fastapi" not in sys.modules:
    try:
        import fastapi  # type: ignore
    except ImportError:
        fastapi_mod = ModuleType("fastapi")
        responses_mod = ModuleType("fastapi.responses")
        concurrency_mod = ModuleType("fastapi.concurrency")

        class HTTPException(Exception):
            def __init__(self, status_code: int, detail: str = ""):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        class FastAPI:
            def __init__(self, **kwargs):
                self.routes = []

            def get(self, path, **kwargs):
                def decorator(fn):
                    return fn
                return decorator

            def post(self, path, **kwargs):
                def decorator(fn):
                    return fn
                return decorator

            def delete(self, path, **kwargs):
                def decorator(fn):
                    return fn
                return decorator

        def Body(*args, **kwargs):
            return None

        class FileResponse:
            def __init__(self, path: str, filename: str = "", media_type: str = ""):
                self.path = path
                self.filename = filename
                self.media_type = media_type

        async def run_in_threadpool(func, *args, **kwargs):
            return func(*args, **kwargs)

        fastapi_mod.FastAPI = FastAPI
        fastapi_mod.HTTPException = HTTPException
        fastapi_mod.Body = Body
        responses_mod.FileResponse = FileResponse
        concurrency_mod.run_in_threadpool = run_in_threadpool

        sys.modules["fastapi"] = fastapi_mod
        sys.modules["fastapi.responses"] = responses_mod
        sys.modules["fastapi.concurrency"] = concurrency_mod

if "httpx" not in sys.modules:
    try:
        import httpx  # type: ignore
    except ImportError:
        httpx_mod = ModuleType("httpx")
        class AsyncClient:
            def __init__(self, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
        httpx_mod.AsyncClient = AsyncClient
        sys.modules["httpx"] = httpx_mod

if "uvicorn" not in sys.modules:
    try:
        import uvicorn  # type: ignore
    except ImportError:
        uvicorn_mod = ModuleType("uvicorn")
        uvicorn_mod.run = lambda *args, **kwargs: None
        sys.modules["uvicorn"] = uvicorn_mod
