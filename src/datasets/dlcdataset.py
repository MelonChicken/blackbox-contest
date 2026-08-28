from pathlib import Path
from torch.utils.data import Dataset
from src.utils import clip

class DLCStage1Dataset(Dataset):
    def __init__(
        self,
        dataframe,
        data_root,
        frames=16,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.frames = frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        video_path = (
            self.data_root
            / row["path"]
        )

        x = clip(
            video_path,
            self.frames,
        )

        y = int(row["label"])

        return x, y