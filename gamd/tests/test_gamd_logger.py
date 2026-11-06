from unittest.mock import MagicMock

from gamd.GamdLogger import GamdLogger


def test_log_line_reaches_disk_without_closing(tmp_path):
    """A job killed at its time limit never closes the log, so each line must be
    on disk as soon as it is written."""
    log_path = tmp_path / "gamd.log"
    tracked_value = MagicMock()
    for method in ("get_reporting_starting_energy",
                   "get_reporting_force_scaling_factor",
                   "get_reporting_boost_potential",
                   "get_reporting_effective_harmonic_constant"):
        getattr(tracked_value, method).return_value = "1.0"

    logger = GamdLogger.__new__(GamdLogger)
    logger.gamdLog = open(log_path, "w")
    logger.tracked_values = [tracked_value, tracked_value]

    logger.write_to_gamd_log(500)

    assert "\t500\t" in log_path.read_text()
    logger.close()
