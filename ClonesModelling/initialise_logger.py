import logging
import os


def initialise_logger(
    name: str,
    logging_directory: str = "logs",
    sub_folder: str | None = None,
    file_logging_level: int = logging.DEBUG,
    console_logging_level: int = logging.INFO,
    mode: str = "w+",
    additional_logger_to_file_handle: logging.Logger | None = None,
) -> logging.Logger:
    logger_name, log_file_path = parse_log_folder(name, logging_directory, sub_folder)

    logger = logging.getLogger(logger_name)
    remove_handlers(log_file_path, logger)

    logger.setLevel(logging.DEBUG)  # to defer to handler levels

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_logging_level)
    console_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(log_file_path, mode)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(file_logging_level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    if additional_logger_to_file_handle is not None:
        additional_logger_to_file_handle.addHandler(file_handler)
    return logger


def initialise_csv_logger(
    name: str,
    titles: list[str],
    logging_directory: str = "logs",
    sub_folder: str | None = None,
):
    logger_name, log_file_path = parse_log_folder(
        name, logging_directory, sub_folder, ".csv"
    )
    csv_logger = logging.getLogger(logger_name)
    remove_handlers(log_file_path, csv_logger)

    file_handler = logging.FileHandler(log_file_path, "w+")
    csv_formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(csv_formatter)
    csv_logger.addHandler(file_handler)
    csv_logger.setLevel(logging.INFO)

    csv_logger.info(",".join(titles))
    return csv_logger


def parse_log_folder(
    name: str, logging_directory: str, sub_folder: str | None, extension: str = ".log"
) -> tuple[str, str]:
    sub_folder_slash = (sub_folder + "/") if sub_folder is not None else ""
    os.makedirs(f"{logging_directory}/{sub_folder_slash}", exist_ok=True)

    log_file_path = f"{logging_directory}/{sub_folder_slash}{name}{extension}"
    logger_name = f"{sub_folder_slash}{name}"
    return logger_name, log_file_path


def remove_handlers(log_file_path, logger):
    if logger.hasHandlers():
        for handler in logger.handlers:
            handler.close()
        logger.handlers = []

        if os.path.isfile(log_file_path):
            with open(log_file_path, "r", encoding="utf-8"):
                # deletes what was previously in the file, unless mode is append
                pass
