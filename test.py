from network import *

config = {'network_folder':'net_topos', 'network':'net4.txt', 'network_bandwidth':10, 'network_delay_low':5, 'network_delay_high':5, \
          'number_of_flows_range':range(1, 50, 5), 'duplicates':10, \
          'flow_bandwidth_low':3, 'flow_bandwidth_high':5, 'flow_delay_low':0, 'flow_delay_high':200, \
          'same_flow_test':True, 'file_output':False, 'output_folder':'outputs', 'file_name_prefix':'same_flow_test' \
         }

run_experiment(config)
