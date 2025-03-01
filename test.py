import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)  # (rows, cols, index)
plt.plot(x, y1, label="sin(x)")
plt.legend()
plt.title("Sine Wave")

plt.subplot(1, 2, 2)
plt.plot(x, y2, label="cos(x)", color="red")
plt.legend()
plt.title("Cosine Wave")

plt.show()