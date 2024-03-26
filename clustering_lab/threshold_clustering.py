import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))
def threshold_clustering(data, threshold):
    clusters = []
    cluster_indices = [] # Список для хранения индексов кластеров для каждой точки
    for point in data:
        # Инициализация кластера для текущей точки
        new_cluster = [point]
        cluster_to_update = None # Добавляем переменную для отслеживания кластера, который нужно обновить
        for i, cluster in enumerate(clusters):
            # Проверка, находится ли текущая точка внутри порогового расстояния от любой точки в кластере
            if any(euclidean_distance(point, cluster_point) <= threshold for cluster_point in cluster):
                new_cluster = cluster + [point]
                cluster_to_update = i # Отмечаем индекс кластера для обновления
                break
        # Если точка не была добавлена в существующий кластер, создаем новый кластер
        if new_cluster == [point]:
            clusters.append(new_cluster)
            cluster_indices.append(len(clusters) - 1) # Добавляем индекс нового кластера
        else:
            # Обновляем кластер в списке кластеров
            if cluster_to_update is not None:
                clusters[cluster_to_update] = new_cluster
                cluster_indices.append(cluster_to_update) # Добавляем индекс обновленного кластера
    return cluster_indices
