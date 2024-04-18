from .zjumocap import ZJUMoCapDataset
from .people_snapshot import PeopleSnapshotDataset
from .X_Humans import X_HumansDataset

def load_dataset(cfg, split='train'):
    dataset_dict = {
        'zjumocap': ZJUMoCapDataset,
        'people_snapshot': PeopleSnapshotDataset,
        'X_Humans': X_HumansDataset
    }
    return dataset_dict[cfg.name](cfg, split)
