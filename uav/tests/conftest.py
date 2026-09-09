def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: tests taking more than 30 seconds"
    )
