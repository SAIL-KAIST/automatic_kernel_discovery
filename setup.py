from distutils.core import setup

setup(name="Kernel Discovery",
      version="0.1",
      author="Anh Tong",
      packages=["kernel_discovery",
                "kernel_discovery.analysis", 
                "kernel_discovery.description",
                "kernel_discovery.expansion",
                "kernel_discovery.evaluation",
                "kernel_discovery.sparse_selector",
                "kernel_discovery.plot",
                "kernel_discovery.io"])