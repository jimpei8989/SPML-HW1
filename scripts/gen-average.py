lines = []

while True:
    try:
        lines.append(input())
    except EOFError:
        break

lines = [l.replace(r'\\', '').replace(r'\textbf{', '').replace(r'}', '') for l in lines]
lines = [list(map(float, l.split('&')[2:])) for l in lines]

N, M = len(lines), len(lines[0])

print(' & '.join(map(lambda k: f'{k[1]:.2f}' if k[0] < M - 2 else f'{k[1]:.4f}', enumerate(sum(lines[i][j] for i in range(N)) / N  for j in range(M)))))

