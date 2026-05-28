# Flask Configuration

class Config:
    """Base configuration"""
    SECRET_KEY = 'your-secret-key-change-this'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
