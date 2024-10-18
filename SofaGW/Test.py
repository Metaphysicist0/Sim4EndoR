from SofaGW import SimController, example_vessel
from SofaGW.utils import SaveImage, root_dir


def test_installation():
    sim = SimController(timeout=10000,
                        vessel_filename=example_vessel)
    errclose = True
    for i in range(1500):
        sim.action(translation=1000, rotation=50)
        sim.action(translation=300, rotation=20)
        sim.action(translation=100, rotation=50)
        sim.action(translation=400, rotation=-20)
        errclose = sim.step(realtime=False)
        if errclose:
            sim.reset()

# 运行测试函数
test_installation()