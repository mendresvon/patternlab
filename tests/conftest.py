"""Make the reference website tests use local test fixtures."""
import models.service as service
from tests.fakes import FakeGenerator, FakePredictor

service.FakePredictor = FakePredictor
service.FakeGenerator = FakeGenerator
