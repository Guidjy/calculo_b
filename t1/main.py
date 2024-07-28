import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from tabulate import tabulate


def main():
    x = sp.Symbol('x')
    y = sp.Symbol('y')

    # sistema de equações
    f = x**2 + y**2 - 1
    g = y - sp.sin(x)

    fxy = sp.lambdify((x, y), f, 'numpy')
    gxy = sp.lambdify((x, y), g, 'numpy')

    #derivadas parciais das equações
    df_dx = sp.diff(f, x)
    df_dx = sp.lambdify((x, y), df_dx, 'numpy')
    df_dy = sp.diff(f, y)
    df_dy = sp.lambdify((x, y), df_dy, 'numpy')
    dg_dx = sp.diff(g, x)
    dg_dx = sp.lambdify((x, y), dg_dx, 'numpy')
    dg_dy = sp.diff(g, y)
    dg_dy = sp.lambdify((x, y), dg_dy, 'numpy')

    #chute inicial
    chute_inicial = [2,3]

    # matriz jacobiana
    mat_jacob = np.array([[df_dx, df_dy], 
                          [dg_dx, dg_dy]])

    # número de iterações
    n_iteracoes = 0
    # histórico das iterações
    historico = {'n': [], 'xn': [], 'yn': [], 'f(xn, yn)': [], 'g(xn, yn)': []}

    # repete até achar uma solução para o sistema ou 1000 vezes
    xn = chute_inicial[0]
    yn = chute_inicial[1]
    while n_iteracoes < 1000:
        
        # avalia a matriz Jacobiana no ponto atual
        mat_jacob = np.array([[df_dx(xn, yn), df_dy(xn, yn)], 
                              [dg_dx(xn, yn), dg_dy(xn, yn)]])
        # descobre a matriz inversa
        inversa = np.linalg.inv(mat_jacob)

        # descobre Δx e Δy
        delta_x = (-inversa[0][0]) * fxy(xn, yn) + (-inversa[0][1]) * gxy(xn, yn)
        delta_y = (-inversa[1][0]) * fxy(xn, yn) + (-inversa[1][1]) * gxy(xn, yn)

        # calcula xn+1 e yn+1
        xn1 = xn + delta_x
        yn1 = yn + delta_y

        # registra os dados da iteração no histórico
        historico['n'].append(n_iteracoes)
        historico['xn'].append(round(xn, 3))
        historico['yn'].append(round(yn, 3))
        historico['f(xn, yn)'].append(round(fxy(xn, yn), 3))
        historico['g(xn, yn)'].append(round(gxy(xn, yn), 3))

        # verifica se chegou a uma solução
        if abs(delta_x) < 0.0001 and abs(delta_y) < 0.0001:
            break

        # atualiza os dados para a próxima iteração
        xn = xn1
        yn = yn1
        n_iteracoes += 1
    
    # mostra os resultados
    print(tabulate(historico, headers='keys', tablefmt='grid'))
    print('>>> Soluções: ')
    print(f'> x = {round(xn, 3)}')
    print(f'> y = {round(yn, 3)}')
    plota_grafo(fxy, gxy, historico)


def plota_grafo(fxy, gxy, historico):
    x = sp.Symbol('x')
    y = sp.Symbol('y')

    # criar uma grade de valores de x e y
    x_vals = np.linspace(-2, 2, 400)
    y_vals = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x_vals, y_vals)

    # calcular os valores de f e g na grade
    F = fxy(X, Y)
    G = gxy(X, Y)

    plt.figure(figsize=(12, 6))

    # plotar a equação f(x, y) = 0
    plt.contour(X, Y, F, levels=[0], colors='blue')
    # plotar a equação g(x, y) = 0
    plt.contour(X, Y, G, levels=[0], colors='red')

    # plotar os pontos de cada iteração
    plt.plot(historico['xn'], historico['yn'], 'go-', label='Iterações')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gráficos das equações f(x, y) = 0 e g(x, y) = 0 com pontos de iteração')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend(['x^2 + y^2 - 1 = 0', 'y - sin(x) = 0', 'Iterações'])
    plt.show()


if __name__ == '__main__':
    main()
