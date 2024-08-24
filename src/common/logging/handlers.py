# import logging
# from logging.handlers import RotatingFileHandler
# from pathlib import Path
#
# from kedro.framework.project import configure_logging
#
#
# def configurate_kedro_logging(
#     log_path: Path,
#     logging_level: str = "INFO",
#     backup_bytes: int = 1000000,
#     backup_count: int = 3,
# ) -> None:
#     project_logger_config = {
#         "version": 1,
#         "root": {"handlers": ["file_handler"], "level": logging_level},
#         "handlers": {
#             # "console": {
#             #     "class": "logging.StreamHandler",
#             #     "level": logging_level,
#             #     "formatter": "default",
#             # },
#             "file_handler": {
#                 "class": "logging.handlers.RotatingFileHandler",
#                 "level": "DEBUG",
#                 "formatter": "default",
#                 "filename": str(log_path.joinpath("kedro.log")),
#                 "maxBytes": backup_bytes,
#                 "backupCount": backup_count,
#             },
#         },
#         "formatters": {
#             "default": {
#                 "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
#                 "datefmt": "%d-%b-%y %H:%M:%S",
#             }
#         },
#     }
#     configure_logging(project_logger_config)
#
#
# _FORMATTER = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "[%d-%b-%y %H:%M:%S]")
#
#
# def get_file_handler(log_filename: str) -> logging.Handler:
#     handler = RotatingFileHandler(
#         filename=log_filename,
#         maxBytes=1000000,
#         backupCount=3,
#     )
#     handler.setFormatter(_FORMATTER)
#     return handler
#
#
# def get_stream_handler() -> logging.Handler:
#     handler = logging.StreamHandler()
#     handler.setFormatter(_FORMATTER)
#     handler.setLevel("INFO")
#     return handler
#
#
# def set_project_logging(top_name: str, logs_path) -> None:
#     stream_handler = get_stream_handler()
#     project_file_handler = get_file_handler(str(Path(logs_path, "project.log")))
#
#     top_logger = logging.getLogger(top_name)
#     top_logger.addHandler(stream_handler)
#     top_logger.addHandler(project_file_handler)
#
#     configurate_kedro_logging(
#         log_path=logs_path,
#     )
