#    dl-classification
#    Copyright (C) 2017  Matthieu Ospici
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


from pathlib import Path


class file_to_label_binary:
    def __init__(self):
        self.classes = {}
        self.curr_class = 0

    def to_label(self, in_file):
        p = Path(in_file)

        f_dir = p.parts[-2]

        if f_dir not in self.classes:
            self.classes[f_dir] = self.curr_class
            self.curr_class += 1

            # assert self.curr_class < 3,"For now, only two classes are supported (binary classification)"

        return self.classes[f_dir]
