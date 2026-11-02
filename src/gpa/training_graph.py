"""Static-shape forward/backward replay; optimizer remains outside capture.

Warmups do NOT update parameters. Parameters and input/label buffers retain
their addresses. Replay overwrites the captured leaf gradient buffers; never
set .grad=None after capture while reusing a graph.
"""
import torch


class GradientGraph:
    def __init__(self, model, inputs, loss_fn, warmups=3):
        self.inputs = tuple(x.detach().clone().requires_grad_(x.requires_grad) for x in inputs)
        current = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(warmups):
                model.zero_grad(set_to_none=True)
                for x in self.inputs:
                    x.grad = None
                loss, output = loss_fn(*self.inputs)
                loss.backward()
        current.wait_stream(stream)
        model.zero_grad(set_to_none=True)
        for x in self.inputs:
            x.grad = None
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss, self.output = loss_fn(*self.inputs)
            self.loss.backward()

    def replay(self, *inputs):
        with torch.no_grad():
            for static, value in zip(self.inputs, inputs):
                if static.shape != value.shape or static.dtype != value.dtype:
                    raise ValueError("CUDA Graph requires the captured input shapes/dtypes")
                static.copy_(value)
        self.graph.replay()
        return self.loss, self.output
