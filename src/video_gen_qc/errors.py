"""Runtime failures are errors, never evidence of poor video quality."""


class VideoQCError(Exception):
    """An actionable input, configuration, provider, or pipeline error."""


class InputError(VideoQCError):
    pass


class ConfigError(VideoQCError):
    pass


class ProviderError(VideoQCError):
    pass


class VideoError(VideoQCError):
    pass


class JudgmentError(VideoQCError):
    pass


class OutputError(VideoQCError):
    pass
