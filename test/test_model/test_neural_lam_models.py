#  from dlam.model.neural_lam_model.ar_model import ARModel
#
#
#  def test_instantiate_hilam():
#      kwargs = {
#          "num_state_vars": 1,
#          "num_forcing_vars": 1,
#          "num_past_forcing_steps": 1,
#          "num_future_forcing_steps": 1,
#          "da_state_stats": 1,
#          "da_boundary_mask": 1,
#      }
#
#      ar_model = ARModel(**kwargs)


#  num_state_vars = datastore.get_num_data_vars(category="state")
#  num_forcing_vars = datastore.get_num_data_vars(category="forcing")
#  # Load static features standardized
#  da_static_features = datastore.get_dataarray(
#      category="static", split=None, standardize=True
#  )
#  da_state_stats = datastore.get_standardization_dataarray(category="state")
#  da_boundary_mask = datastore.boundary_mask
#  weather_dataset = WeatherDataset(datastore=self._datastore, split=split)
