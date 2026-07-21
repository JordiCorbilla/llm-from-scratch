from frankengpt.cli import build_parser


def test_root_help_lists_every_supported_workflow():
    help_text = build_parser().format_help()
    for command in (
        "train",
        "generate",
        "benchmark",
        "fetch-data",
        "finetune-pretrained",
        "generate-pretrained",
    ):
        assert command in help_text
