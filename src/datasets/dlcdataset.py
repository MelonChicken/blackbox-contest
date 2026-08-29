from pathlib import Path
from torch.utils.data import Dataset
from src.utils import clip


class DLCStage1Dataset(Dataset):
    """
    DLC-2021 Stage 1 Dataset

    ORIGINAL   -> 0
    RERECORDED -> 1
    """

    def __init__(
        self,
        dataframe,
        data_root,
        frames,
        mean,
        std,
    ):
        # dataframe index를 0부터 다시 정리한다.
        self.df = dataframe.reset_index(drop=True)

        # DLC-2021 root 경로를 저장한다.
        self.data_root = Path(data_root)

        # 영상 하나에서 추출할 frame 수를 저장한다.
        self.frames = frames

        # Stage 1 normalization에 사용할 mean/std를 저장한다.
        self.mean = mean
        self.std = std


    def __len__(self):
        # 전체 sample 개수를 반환한다.
        return len(self.df)


    def __getitem__(self, index):
        # index에 해당하는 dataframe row를 가져온다.
        row = self.df.iloc[index]

        # CSV의 상대경로와 DLC root를 결합하여
        # 실제 영상 경로를 만든다.
        video_path = (
            self.data_root
            / row["path"]
        )

        # 영상에서 지정한 수의 frame을 sampling한다.
        x, _ = clip(
            video_path,
            self.frames,
        )

        # Stage 1 normalization을 적용한다.
        x = (
            x - self.mean
        ) / self.std

        # 문자열 label을 정수 label로 변환한다.
        y = (
            0
            if row["label"] == "ORIGINAL"
            else 1
        )

        # x: [C, T, H, W]
        # y: int
        return x, y