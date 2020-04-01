config = {'network_folder':'net_topos', 'network':'net14.txt', 'network_bandwidth':10, 'network_delay_low':5, 'network_delay_high':5, \
          'number_of_flows_range':range(1, 202, 10), 'duplicates':10, \
          'flow_bandwidth_low':3, 'flow_bandwidth_high':5, 'flow_delay_low':0, 'flow_delay_high':200, \
          'same_flow_test':False, 'file_output':True, 'output_folder':'outputs', 'file_name_prefix':''  \
         }



# normal
config = {'network_folder':'net_topos', 'network':'net50.txt', 'network_bandwidth':10, 'network_delay_low':5, 'network_delay_high':5, \
          'number_of_flows_range':range(1, 202, 10), 'duplicates':5, \
          'flow_bandwidth_low':3, 'flow_bandwidth_high':5, 'flow_delay_low':0, 'flow_delay_high':200, \
          'same_flow_test':False, 'file_output':True, 'output_folder':'outputs', 'file_name_prefix':''  \
         }


# low_flow_number_
config = {'network_folder':'net_topos', 'network':'net50.txt', 'network_bandwidth':10, 'network_delay_low':5, 'network_delay_high':5, \
          'number_of_flows_range':range(1, 32, 2), 'duplicates':5, \
          'flow_bandwidth_low':3, 'flow_bandwidth_high':5, 'flow_delay_low':0, 'flow_delay_high':200, \
          'same_flow_test':False, 'file_output':True, 'output_folder':'outputs', 'file_name_prefix':'low_flow_number_'  \
         }