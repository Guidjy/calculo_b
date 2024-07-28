import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate


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



def main():
    
    # função para derivação
    x = sp.Symbol('x')
    y = sp.Symbol('y')

    z = x**2 + x*y + 10*y**2 - 5*x - 3*y

    # função para cálculo
    fxy = sp.lambdify((x, y), z, 'numpy')

    # derivadas parciais da função
    df_dx = sp.diff(z, x)
    fx = sp.lambdify((x, y), df_dx, 'numpy')
    df_dy = sp.diff(z, y)
    fy = sp.lambdify((x, y), df_dy, 'numpy')

    # vetor gradiente
    gradiente = [fx, fy]

    # passo da descida de gradiente
    passo = 0.01

    # chute inicial  x  y
    chute_inicial = [3, 5]

    # histórico das iterações
    historico = {'n': [], 'xn': [], 'yn': [], 'f(xn, yn)': []}
    # número de iterções
    n_iteracoes = 0

    # repete até achar um ponto de mínimo ou 10000 vezes
    xn = chute_inicial[0]
    yn = chute_inicial[1]
    while n_iteracoes < 10000:
        # (xn+1, yn+1) = (xn, yn) - h * ∇f(xn, yn)
        xn1 = xn - passo * fx(xn, yn)
        yn1 = yn - passo * fy(xn, yn)

        # registra os dados da iteração no histórico
        historico['n'].append(n_iteracoes)
        historico['xn'].append(xn)
        historico['yn'].append(yn)
        historico['f(xn, yn)'].append(fxy(xn, yn))

        # se o módulo da diferença de f(xn, yn) e f(xn+1, yn+1) for menor do que 0.01, achou um ponto de mínimo local
        if np.abs(fxy(xn1, yn1) - fxy(xn, yn)) < 0.000001:
            break

        # atualiza os dados para a próxima iteração
        xn = xn1
        yn = yn1
        n_iteracoes += 1
    
    minimo_local = [round(xn, 1), round(yn, 1)]
    print(tabulate(historico, headers='keys', tablefmt='grid'))
    print(f'ponto de mínimo local: {minimo_local}')

    plota_grafo(fxy, historico)


if __name__ == '__main__':
    main()
