import matplotlib.pyplot as plt

from results import *

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.plot(q_bs32_ds4_ep100[1], label='quantum - 4x4 input size', c='b')
plt.plot(c_bs32_ds4_ep100[1], label='classical - 4x4 input size', c='r')
plt.plot(q_bs32_ds3_ep100[1], label='quantum - 3x3 input size', c='b', linestyle=':')
plt.plot(c_bs32_ds3_ep100[1], label='classical - 3x3 input size', c='r', linestyle=':')
plt.legend()
plt.title('Train accuracy')

plt.subplot(2, 2, 2)
plt.plot(q_bs32_ds4_ep100[3], label='quantum - 4x4 input size', c='b')
plt.plot(c_bs32_ds4_ep100[3], label='classical - 4x4 input size', c='r')
plt.plot(q_bs32_ds3_ep100[3], label='quantum - 3x3 input size', c='b', linestyle=':')
plt.plot(c_bs32_ds3_ep100[3], label='classical - 3x3 input size', c='r', linestyle=':')
plt.legend()
plt.title('Test accuracy')

plt.subplot(2, 2, 3)
plt.plot(q_bs32_ds4_ep100[0], label='quantum - 4x4 input size', c='b')
plt.plot(c_bs32_ds4_ep100[0], label='classical - 4x4 input size', c='r')
plt.plot(q_bs32_ds3_ep100[0], label='quantum - 3x3 input size', c='b', linestyle=':')
plt.plot(c_bs32_ds3_ep100[0], label='classical - 3x3 input size', c='r', linestyle=':')
plt.legend()
plt.title('Train loss')

plt.subplot(2, 2, 4)
plt.plot(q_bs32_ds4_ep100[2], label='quantum - 4x4 input size', c='b')
plt.plot(c_bs32_ds4_ep100[2], label='classical - 4x4 input size', c='r')
plt.plot(q_bs32_ds3_ep100[2], label='quantum - 3x3 input size', c='b', linestyle=':')
plt.plot(c_bs32_ds3_ep100[2], label='classical - 3x3 input size', c='r', linestyle=':')
plt.legend()
plt.title('Test loss')
plt.tight_layout()

plt.savefig('results_2kernels.png', dpi=300)
