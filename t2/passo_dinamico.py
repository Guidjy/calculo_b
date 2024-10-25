import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from tabulate import tabulate

def main():
    # Função para derivação
    x, y = sp.symbols('x y')
    z = x**2 + x*y + 10*y**2 - 5*x - 3*y

    # Função para cálculo
    fxy = sp.lambdify((x, y), z, 'numpy')

    # Derivadas parciais da função
    df_dx = sp.diff(z, x)
    fx = sp.lambdify((x, y), df_dx, 'numpy')
    df_dy = sp.diff(z, y)
    fy = sp.lambdify((x, y), df_dy, 'numpy')

    # Passo inicial e parâmetros de ajuste dinâmico
    passo = 0.1
    alpha = 0.5  # fator de redução do passo
    beta = 0.9   # taxa de ajuste (entre 0 e 1 para diminuir o passo se necessário)

    # Chute inicial
    chute_inicial = [3, 5]
    historico = {'n': [], 'xn': [], 'yn': [], 'f(xn, yn)': []}
    n_iteracoes = 0

    # Definindo xn e yn
    xn, yn = chute_inicial

    while n_iteracoes < 10000:
        # Gradiente e cálculo do próximo ponto
        grad_x = fx(xn, yn)
        grad_y = fy(xn, yn)
        xn1 = xn - passo * grad_x
        yn1 = yn - passo * grad_y

        # Critério de parada baseado na diferença da função objetivo
        if np.abs(fxy(xn1, yn1) - fxy(xn, yn)) < 0.000001:
            break

        # Ajuste do passo se a função não reduzir o suficiente
        if fxy(xn1, yn1) > fxy(xn, yn):
            passo *= alpha  # reduz o passo
        else:
            passo *= beta   # ajusta levemente o passo

        # Registro e atualização
        historico['n'].append(n_iteracoes)
        historico['xn'].append(xn)
        historico['yn'].append(yn)
        historico['f(xn, yn)'].append(fxy(xn, yn))

        xn, yn = xn1, yn1
        n_iteracoes += 1

    # Resultados
    minimo_local = [round(xn, 3), round(yn, 3)]
    print(tabulate(historico, headers='keys', tablefmt='grid'))
    print(f'\n>>>ponto de mínimo local: {minimo_local}')
    plota_grafo(fxy, historico)

def plota_grafo(fxy, hist):
    # Generate x and y values
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)

    # Create a meshgrid for x and y values
    X, Y = np.meshgrid(x, y)
    Z = fxy(X, Y)

    # Create a figure with two subplots: one for the 3D plot and one for the contour plot
    fig = plt.figure(figsize=(14, 6))

    # pontos x, y e z
    pontos_x = hist['xn']
    pontos_y = hist['yn']
    pontos_z = hist['f(xn, yn)']

    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    ax1.set_title('3D Surface Plot')
    # Create a scatter plot in 3D with red points
    ax1.scatter(pontos_x, pontos_y, pontos_z, color='red')

    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, cmap='viridis')
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_title('Contour Plot')
    fig.colorbar(contour, ax=ax2, shrink=0.8, aspect=5) # Add a colorbar to the contour plot
    ax2.scatter(pontos_x, pontos_y, color='red')

    # Show the plots
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
