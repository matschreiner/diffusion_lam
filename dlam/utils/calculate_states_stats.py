import torch


def calculate_state_stats(datastore):
    num_state_vars = datastore.get_num_data_vars(category="state")
    num_forcing_vars = datastore.get_num_data_vars(category="forcing")
    da_state_stats = datastore.get_standardization_dataarray(category="state")
    da_boundary_mask = datastore.boundary_mask

    state_stats = {
        "state_mean": torch.tensor(
            da_state_stats.state_mean.values, dtype=torch.float32
        ),
        "state_std": torch.tensor(da_state_stats.state_std.values, dtype=torch.float32),
        # Note that the one-step-diff stats (diff_mean and diff_std) are
        # for differences computed on standardized data
        "diff_mean": torch.tensor(
            da_state_stats.state_diff_mean_standardized.values,
            dtype=torch.float32,
        ),
        "diff_std": torch.tensor(
            da_state_stats.state_diff_std_standardized.values,
            dtype=torch.float32,
        ),
    }
