from clustering_lab import kmeans, threshold_clustering, dirichlet_process_mixture
from clustering_lab import load_dataset

import matplotlib.pyplot as plt

from os import mkdir, chdir

DISTRIBUTIONS = [
        'circular',
        'elongated',
        'g-shaped',
        'elongated-diff_sizes',
        'g-shaped-closer',
        'circular-varying_prox',
        'elongated-very_close',
        'g-shaped-varying_sizes',
        'circular-mixed_sizes-proximity'
    ]

def init():


    chdir('reports')
    for _ in DISTRIBUTIONS:
        mkdir(_)



def create_report_for_dataset(dataset_name: str, params_for_algo: dict)-> None:
    data, correct_clusters = load_dataset(dataset_name)

    plt.scatter(data[:, 0], data[:, 1], c=correct_clusters, s=20, cmap='viridis')

    # Добавляем заголовки осей и заголовок графика
    plt.xlabel('Ось X')
    plt.ylabel('Ось Y')
    plt.title(dataset_name)

    # Сохраняем изображение в файл
    plt.savefig('reports/' + dataset_name + "/correct.png")

    # Закрываем график, чтобы избежать его отображения
    plt.close()


    kmeans_clusters, kmeans_centers = kmeans(data, *params_for_algo["kmeans"])

    plt.scatter(data[:, 0], data[:, 1], c=kmeans_clusters, s=20, cmap='viridis')

    # Добавляем заголовки осей и заголовок графика
    plt.xlabel('Ось X')
    plt.ylabel('Ось Y')
    plt.title("kmeans")

    # Сохраняем изображение в файл
    plt.savefig('reports/' + dataset_name + "/kmeans.png")

    # Закрываем график, чтобы избежать его отображения
    plt.close()



    threshold_clustering_clusters = threshold_clustering(data, *params_for_algo["threshold_clustering"])

    plt.scatter(data[:, 0], data[:, 1], c=threshold_clustering_clusters, s=20, cmap='viridis')

    plt.xlabel('Ось X')
    plt.ylabel('Ось Y')
    plt.title("threshold_clustering")
    plt.savefig('reports/' + dataset_name + "/threshold_clustering.png")
    plt.close()

    dirichlet_clusters, probabilitys = dirichlet_process_mixture(data, *params_for_algo["dirichlet"])

    plt.scatter(data[:, 0], data[:, 1], c=dirichlet_clusters, s=20, cmap='viridis')

    plt.xlabel('Ось X')
    plt.ylabel('Ось Y')
    plt.title("dirichlet")
    plt.savefig('reports/' + dataset_name + "/dirichlet.png")
    plt.close()

    with open('reports/' + dataset_name + "/report.txt", 'w', encoding='utf-8') as file:
        file.write("Число реальных (подразумеваемых) кластеров: "+str(1+max(correct_clusters)))

        file.write("\nЧисло кластеров для kmean: "+str(*params_for_algo["kmeans"]))
        file.write('\nЦентры в алгоритме kmean [задается]:\n')
        file.write(str(kmeans_centers))
        file.write("\nрасстояние для threshold_clustering: "+str(*params_for_algo["threshold_clustering"]))
        file.write("\nЧисло кластеров для threshold_clustering [определено]: "+str(1+max(threshold_clustering_clusters)))
        file.write("\nЧисло кластеров для Дирихле [задается]: "+str(params_for_algo["dirichlet"][0]))




def main():
    params = {"kmeans": [3], "threshold_clustering": [2], "dirichlet": [3, 1/300]}

    for _ in DISTRIBUTIONS:
        create_report_for_dataset(_, params)

def test():
    params = {"kmeans": [3], "threshold_clustering": [2], "dirichlet": [3, 1 / 300]}
        #'circular',
        #'elongated',
        #'g-shaped',
        #'elongated-diff_sizes',
        #'g-shaped-closer',
        #'circular-varying_prox',
        #'elongated-very_close',
        #'g-shaped-varying_sizes',
        #'circular-mixed_sizes-proximity'
    create_report_for_dataset('circular', params)

main()
