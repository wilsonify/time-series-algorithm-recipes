LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,  # Keeps other loggers active
        'formatters': {
            'default': {
                'format': '%(asctime)s - %(levelname)s - %(message)s',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'DEBUG',
                'formatter': 'default',
                'stream': 'ext://sys.stdout',  # or 'ext://sys.stderr'
            },
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['console'],
        },
    }