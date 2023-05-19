import matplotlib.pyplot as plt

from results import *

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.plot(q_bs32_ds4_ep100[1], label='q_bs32_ds4_ep100')
plt.plot(q_bs32_ds3_ep100[1], label='q_bs32_ds3_ep100')
plt.plot(c_bs32_ds4_ep100[1], label='c_bs32_ds4_ep100')
plt.plot(c_bs32_ds3_ep100[1], label='c_bs32_ds3_ep100')
plt.legend()
plt.title('train accuracy')

plt.subplot(2, 2, 2)
plt.plot(q_bs32_ds4_ep100[3], label='q_bs32_ds4_ep100')
plt.plot(q_bs32_ds3_ep100[3], label='q_bs32_ds3_ep100')
plt.plot(c_bs32_ds4_ep100[3], label='c_bs32_ds4_ep100')
plt.plot(c_bs32_ds3_ep100[3], label='c_bs32_ds3_ep100')
plt.legend()
plt.title('test accuracy')

plt.subplot(2, 2, 3)
plt.plot(q_bs32_ds4_ep100[0], label='q_bs32_ds4_ep100')
plt.plot(q_bs32_ds3_ep100[0], label='q_bs32_ds3_ep100')
plt.plot(c_bs32_ds4_ep100[0], label='c_bs32_ds4_ep100')
plt.plot(c_bs32_ds3_ep100[0], label='c_bs32_ds3_ep100')
plt.legend()
plt.title('train loss')

plt.subplot(2, 2, 4)
plt.plot(q_bs32_ds4_ep100[2], label='q_bs32_ds4_ep100')
plt.plot(q_bs32_ds3_ep100[2], label='q_bs32_ds3_ep100')
plt.plot(c_bs32_ds4_ep100[2], label='c_bs32_ds4_ep100')
plt.plot(c_bs32_ds3_ep100[2], label='c_bs32_ds3_ep100')
plt.legend()
plt.title('test loss')
plt.tight_layout()
plt.show()
