import loop_branched
import loop_double
import loop_single

import radiators
import fans
import pumps

def generate_performance_table(loop_module, table_title):
    # Fetch lists of radiators and pumps
    rads = radiators.radiator_list
    pmps = pumps.pump_list
    
    # Extract names for table headers/rows (handling classes or functions gracefully)
    pump_names = [p.__name__ if hasattr(p, '__name__') else str(p) for p in pmps]
    rad_names = [r.__name__ if hasattr(r, '__name__') else str(r) for r in rads]
    
    # Define column widths for text alignment
    first_col_w = 20
    data_col_w = 26
    
    # Print Table Title
    print(f"{'='*50}")
    print(f"{table_title.center(50)}")
    print(f"{'='*50}")
    
    # Print Header Row
    header = f"{'Radiator \\ Pump':<{first_col_w}} | " + " | ".join([f"{name:<{data_col_w}}" for name in pump_names])
    print(header)
    print("-" * len(header))
    
    # Generate and Print Data Rows
    for rad, rad_name in zip(rads, rad_names):
        row_string = f"{rad_name:<{first_col_w}} | "
        cell_strings = []
        
        for pump in pmps:
            # Invoke the using() function with radiator, pump, fan, and boolean flag
            metrics = loop_module.using(rad, pump, fans.Mk11, False)
            
            # Format the 3-element tuple into a single string cell
            # Assuming the metrics are floats; formatted to 2 decimal places
            if table_title == "DOUBLE":
                cell_val = f"({metrics[0][0]:.2f}/{metrics[0][1]:.2f}, {metrics[1]:.2f}, {metrics[2]:.2f})"
            else:
                cell_val = f"({metrics[0]:.2f}, {metrics[1]:.2f}, {metrics[2]:.2f})"
            cell_strings.append(f"{cell_val:<{data_col_w}}")
            
        row_string += " | ".join(cell_strings)
        print(row_string)
        
    print("\n")


if __name__ == "__main__":
    # Create the three tables, one for each type of loop
    generate_performance_table(loop_branched, "BRANCHED")
    generate_performance_table(loop_double, "DOUBLE")
    generate_performance_table(loop_single, "SINGLE")
