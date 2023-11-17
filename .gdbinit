python
import gdb.printing

class StdVectorPrinter:
    "Print a std::vector"

    def __init__(self, val):
        self.val = val

    def to_string(self):
        # Extract vector information
        size = self.val['_M_impl']['_M_finish'] - self.val['_M_impl']['_M_start']
        data_type = self.val.type.template_argument(0)

        # Print the elements
        return "std::vector of size {} with elements of type {}".format(size, data_type)

    def children(self):
        data_start = self.val['_M_impl']['_M_start']
        data_finish = self.val['_M_impl']['_M_finish']

        # Use a generator expression to yield the elements
        return ((str(i), data_start[i]) for i in range(data_finish - data_start))

    def display_hint(self):
        return 'array'

# Register the printer
gdb.printing.register_pretty_printer(None, StdVectorPrinter)
