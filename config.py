import glob
import pathlib
import typing

import pydantic


DATA_DIR: pathlib.Path = pathlib.Path('./data')
REPORT_DIR: pathlib.Path = pathlib.Path('./report')


class Config(pydantic.BaseModel):
    experiment: str = "2024-05--Simulation"

    prompt_classify_files: typing.Dict[str, pathlib.Path] = pydantic.Field(default_factory=lambda: dict())

    raw_data_files: typing.Dict[str, pathlib.Path] = pydantic.Field(default_factory=lambda: dict())
    processed_data_files: typing.Dict[str, pathlib.Path] = pydantic.Field(default_factory=lambda: dict())
    final_data_files: typing.Dict[str, pathlib.Path] = pydantic.Field(default_factory=lambda: dict())

    data_dir: pathlib.Path = None
    report_dir: pathlib.Path = None

    def model_post_init(self, __context: typing.Any) -> None:
        self.data_dir = DATA_DIR / self.experiment
        self.report_dir = REPORT_DIR / self.experiment
        pathlib.Path(self.report_dir).mkdir(parents=True, exist_ok=True)

        self.prompt_classify_files = Config.load_data_dir(f'{DATA_DIR}/prompts_classify/*')

        self.raw_data_files = Config.load_data_dir(f'{DATA_DIR}/{self.experiment}/raw/*')
        self.processed_data_files = Config.load_data_dir(f'{DATA_DIR}/{self.experiment}/processed/*')
        self.final_data_files = Config.load_data_dir(f'{DATA_DIR}/{self.experiment}/final/*')

    @staticmethod
    def load_data_dir(path: str) -> typing.Dict[str, pathlib.Path]:
        return {
            (file_obj := pathlib.Path(file_path)).stem: file_obj
            for file_path in glob.glob(path)
        }
