"""
Utilities for reading and modifying simulation input (.dat) files with updated design parameters.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import datetime
import os

from ensima.classes.logger import Logger


class FileModifier:
    def __init__(self, input_file: str, log_level: str = "info", prefix=""):
        """
        Initialize the FileModifier with input and output file paths.

        Args:
            input_file (str): Path to the input file.
            log_level (str, optional): Log level. Defaults to "debug".
            prefix (str, optional): Log prefix.
        """
        # Initializing the parameters
        self.input_file = input_file
        self.logger = Logger(__name__, level=log_level, prefix=prefix).get()
        self._timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.design_parameters = {}
        self.title_parameters = {}
        self._material = {}
        self._path = os.path.dirname(input_file)
        self._part_name = os.path.splitext(os.path.basename(input_file))[0]
        self._material_file = os.path.join(self._path, self._part_name + ".t51")

        self.get_design_parameter()
        # self.print()

    def get_design_parameter(self) -> None:
        """
        Extract design parameters ("p", "Fr", "db", "Rp") from the input file and store
        them in the 'design_parameters' dictionary. For the thickness ("D"), examine
        the t51 file
        """
        lines = self._get_lines(self.input_file)
        for line_number, line_content in enumerate(lines):
            line_content = line_content.replace(" ", "")
            # Check if the line contains any of the target parameters
            for param in ["Rp", "p", "Fr", "db"]:
                if line_content.startswith(f"$[{param}]="):
                    try:
                        value = float(
                            line_content.split("=")[1]
                        )  # Extract value after '='
                        # Initialize dictionary structure if param not present
                        if param not in self.design_parameters:
                            self.design_parameters[param] = {"values": [], "lines": []}

                        # Store value and line number separately
                        self.design_parameters[param]["values"].append(value)
                        self.design_parameters[param]["lines"].append(line_number)

                        self.logger.debug(
                            f"Found {param} = {value} at line {line_number}"
                        )  # Debug message
                    except ValueError:
                        self.logger.error(
                            f"Warning: Could not parse value for '{param}' on line "
                            f"{line_number}"
                        )  # Handle errors
        # get the thickness
        self.get_blank_thickness()
        self.get_title()

    def _update_title(self):

        parameters = ["Rp", "D", "p", "Fr", "db"]
        values_list = []

        for param in parameters:
            if param not in self.design_parameters:
                val = 0
                self.logger.debug(
                    f"{param} not found in {self.input_file}. Setting to 0."
                )
            else:
                values = self.design_parameters[param].get("values", 0)
                val = values[0] if isinstance(values, list) else values

            try:
                val = f"{float(val):.3f}"
            except (ValueError, TypeError):
                val = str(val)

            values_list.append(val)

        description = " [" + ", ".join(values_list) + "]\n"
        self.title_parameters["values"] = description
        stripped = description.rstrip("\n")
        self.logger.info(f"new title description: {stripped}")

    def set_title(self, lines: list[str]) -> None:
        self._update_title()
        description = self.title_parameters["title"] + self.title_parameters["values"]
        lines[self.title_parameters["line_number"]] = description
        self.logger.debug(f"New title added to {self.input_file}")

    def get_title(self):
        lines = self._get_lines(self.input_file)
        for line_number, line_content in enumerate(lines):
            if line_content.strip().startswith("TITEL"):
                next_line = lines[line_number + 1].strip()
                self.logger.info(f"Original line: {next_line}")

                # Find '[' and ']' to extract parts
                start = next_line.find("[")
                end = next_line.find("]")

                if start != -1 and end != -1 and start < end:
                    # Extract title and values
                    title_text = next_line[:start].strip()
                    values_raw = next_line[start + 1 : end]
                    values = []
                    for v in values_raw.split(","):
                        v = v.strip()
                        if not v:
                            continue
                        if v.lower() == "nan":
                            values.append(float("nan"))
                        else:
                            try:
                                # Try to parse float first
                                values.append(float(v))
                            except ValueError:
                                try:
                                    # If float fails, try int
                                    values.append(int(v))
                                except ValueError:
                                    # If all fail, keep as string
                                    values.append(v)

                    self.title_parameters = {
                        "title": title_text,
                        "values": values,
                        "line_number": line_number + 1,
                    }

                    self.logger.debug(f"Extracted title: {title_text}")
                    self.logger.debug(f"Extracted values: {values}")
                    return
                else:
                    self.logger.warning("No bracketed values found.")
                    self.title_parameters = {
                        "title": next_line.strip(),
                        "values": "",
                        "line_number": line_number + 1,
                    }

    def set_design_parameters(self, parameters: dict) -> None:
        """
        Set design parameters in the input file and write the changes to the output file.

        Args:
            parameters (dict): Dictionary of parameters to set with their new values.
        """
        lines = self._get_lines(self.input_file)
        # create backup
        self._write_lines(
            lines,
            os.path.dirname(self.input_file)
            + "/backup/"
            + os.path.basename(self.input_file)
            + "_"
            + self._timestamp,
        )
        for parameter, value in parameters.items():
            if parameter in self.design_parameters:
                if "Rp" in parameter:
                    self._set_rp(lines, value)
                elif "p" in parameter:
                    self._set_p(lines, value)
                elif "Fr" in parameter:
                    self._set_fr(lines, value)
                elif "db" in parameter:
                    self._set_db(lines, value)
                elif "D" in parameter:
                    self.set_blank_thickness(value)

        # update title:
        self.set_title(lines)
        # write to file
        self._write_lines(lines, self.input_file)

    def _set_p(self, lines: list, value: float) -> None:
        """
        Set the 'p' parameter in the lines.

        Args:
            lines (list): List of lines from the input file.
            value (float): New value for the 'p' parameter.
        """
        line_numbers = self.design_parameters["p"]["lines"]
        old_value = self.design_parameters["p"]["values"][-1]
        for i, line_number in enumerate(line_numbers):
            lines[line_number] = self.replace(lines, line_number, old_value, value)
            lines[line_number + 1] = self.replace(
                lines, line_number + 1, old_value, value, ",", 3
            )
            # update the parameter internally
            self.design_parameters["p"]["values"][i] = value
        self.logger.info(f"p updated to {value}")

    def _set_fr(self, lines: list, value: float) -> None:
        """
        Set the 'Fr' parameter in the lines.

        Args:
            lines (list): List of lines from the input file.
            value (float): New value for the 'Fr' parameter.
        """
        line_numbers = self.design_parameters["Fr"]["lines"]
        old_value = self.design_parameters["Fr"]["values"][-1]
        for i, line_number in enumerate(line_numbers):
            lines[line_number] = self.replace(lines, line_number, old_value, value)
            lines[line_number + 1] = self.replace(
                lines, line_number + 1, old_value, value, ",", 0
            )
            # update the parameter internally
            self.design_parameters["Fr"]["values"][i] = value
        self.logger.info(f"Fr updated to {value}")

    def _set_db(self, lines: list, value: float) -> None:
        """
        Set the 'db' parameter in the lines.

        Args:
            lines (list): List of lines from the input file.
            value (float): New value for the 'db' parameter.
        """
        line_numbers = self.design_parameters["db"]["lines"]
        old_value = self.design_parameters["db"]["values"][-1]
        for i, line_number in enumerate(line_numbers):
            lines[line_number] = self.replace(lines, line_number, old_value, value)
            lines[line_number + 1] = self.replace(
                lines, line_number + 1, old_value, value, ",", 0
            )
            # update the parameter internally
            self.design_parameters["db"]["values"][i] = value
        self.logger.info(f"db updated to {value}")

    def _set_rp(self, lines: list, value: float) -> None:
        """
        Set the 'Rp' parameter in the lines. This requires the Rp files to exist

        Args:
            lines (list): List of lines from the input file.
            value (float): New value for the 'Fr' parameter.
        """
        line_numbers = self.design_parameters["Rp"]["lines"]
        valid_values = self.design_parameters["Rp"]["values"]
        for i, line_number in enumerate(line_numbers):
            if value == valid_values[i]:
                uncomment_line(lines[line_number + 1])
            else:
                comment_line(lines[line_number + 1])

            # update the parameter internally
            self.design_parameters["Rp"]["values"][i] = value
        self.logger.info(f"Rp updated to {value}")

    def get_blank_thickness(self):
        """
        gets the blank thickness in the .t51 file based on given parameter.

        Raises:
            Exception: If the mesh of the blank is corrupted.
        """

        lines = self._get_lines(self._material_file)
        points = []
        for line in lines:
            if "INZIDENZTAFEL" in line:  # Stop reading once we encounter INZIDENZTAFEL
                break
            parts = line.split(",")
            if len(parts) >= 4:
                try:
                    id_point = int(parts[0].strip())
                    x = float(parts[1].strip())
                    y = float(parts[2].strip())
                    z = float(parts[3].strip())
                    points.append((id_point, x, y, z))
                except ValueError:
                    # Skip rows with invalid format
                    continue

        thickness = 0
        z_min, z_max = 0, 0
        seen_points = {}

        for point in points:
            _, x, y, z = point
            key = (x, y)
            if key in seen_points:
                previous_z = seen_points[key]
                z_difference = round(abs(z - previous_z), 10)
                if thickness == 0:
                    thickness = z_difference
                    z_min = min(z, previous_z)
                    z_max = max(z, previous_z)
                else:
                    if thickness != z_difference:
                        raise ValueError("Wrong thickness in file")
            seen_points[key] = z

        self.logger.debug(
            f"Found D = {thickness} in {os.path.basename(self._material_file)}"
        )  # Debug message
        # Calculate old thickness and adjust nodes
        self.design_parameters["D"] = {
            "values": thickness,
            "lines": None,
        }
        self._material = {
            # "points": points,
            "z_min": z_min,
            "z_max": z_max,
        }

    def set_blank_thickness(self, new_thickness):
        """
        gets the blank thickness in the file based on given parameter.

        Args:
            new_thickness (float): The selected parameter value used for the thickness
            calculation.

        Returns:
            list: Updated lines with the thickness adjusted.

        Raises:
            Exception: If the mesh of the blank is corrupted.
        """
        lines = self._get_lines(self._material_file)
        # create backup
        self._write_lines(
            lines,
            os.path.dirname(self._material_file)
            + "/backup/"
            + os.path.basename(self._material_file)
            + "_"
            + self._timestamp,
        )

        midpoint = (self._material["z_max"] + self._material["z_min"]) / 2
        new_z_max = round(midpoint + new_thickness / 2, 6)
        new_z_min = round(midpoint - new_thickness / 2, 6)

        # Check everything is correct
        updated_thickness = round(new_z_max - new_z_min, 6)
        if updated_thickness != round(new_thickness, 6):
            raise ValueError(
                f"Thickness mismatch: design parameter D is {new_thickness}, "
                f"but updated thickness is {updated_thickness}"
            )

        for i, line in enumerate(lines):
            if "INZIDENZTAFEL" in line:  # Stop reading once we encounter INZIDENZTAFEL
                break
            parts = line.split(",")
            if len(parts) >= 4:
                try:
                    id_point = int(parts[0].strip())
                    x = float(parts[1].strip())
                    y = float(parts[2].strip())
                    z = float(parts[3].strip())
                    if z == self._material["z_min"]:
                        z = new_z_min
                    elif z == self._material["z_max"]:
                        z = new_z_max
                    else:
                        raise ValueError("Thickness changed")

                    lines[i] = f"{id_point},  {x:.6e},  {y:.6e},  {z:.6e}\n"
                except ValueError:
                    # Skip rows with invalid format
                    continue

        # write to file
        self._write_lines(lines, self._material_file)
        self.logger.debug(
            f'Changed D = {self.design_parameters["D"]["values"]} --> D = {new_thickness}'
        )
        self.design_parameters["D"]["values"] = new_thickness
        self.logger.info(f"D updated to {new_thickness}")

    def _get_lines(self, file_path: str = "") -> list:
        """
        Reads the file and returns the lines as a list.

        Args:
            file_path (str, optional): Path to the file. Defaults to the input file path.

        Returns:
            list: List of lines from the file.
        """
        if not file_path:
            file_path = self.input_file
        with open(file_path) as file:
            return file.readlines()

    def _write_lines(self, lines: list = None, file_path: str = "") -> None:
        """
        Writes the lines to the specified file.

        Args:
            lines (list): List of lines to write.
            file_path (str, optional): Path to the file. Default to the input file path.
        """
        if not file_path:
            file_path = self.input_file

        directory = os.path.dirname(file_path)
        # Check if the directory exists, and create it if it doesn't
        if directory and not os.path.exists(directory):
            print(f"Creating directory: {directory}")  # Optional: for logging/feedback
            os.makedirs(directory)  # Creates all necessary intermediate directories

        with open(file_path, "w") as file:
            if isinstance(lines, str):
                file.write(lines)
            else:
                file.writelines(lines)

    def print(self):
        """
        Prints the details of the file modifier instance to the console.

        This includes:
        - Input file path
        - Material file path
        - Design parameters
        """
        self.logger.info(f"Input file: {self.input_file}")
        self.logger.info(f"Material file: {self._material_file}")
        self.logger.debug(f"Design parameters:  {self.design_parameters}")

    def __str__(self):
        out = (
            f"Input file: {self.input_file}"
            f"Material file: {self._material_file}"
            f"Design parameters:\n{self.design_parameters}"
        )

        return out

    def __repr__(self):
        return self.__str__()

    def replace(
        self,
        lines: list,
        index: int,
        old_value: float,
        new_value: float,
        delimiter: str = "=",
        pos: int = -1,
    ) -> str:
        """
        Replaces a value in a specific line from a list of lines.

        Args:
            lines (list): The list of strings where the replacement will occur.
            index (int): The index of the line in the list where the replacement will be made.
            old_value (float): The value to be replaced.
            new_value (float): The value to replace with.
            delimiter (str, optional): The delimiter used to split the line. Defaults to "=".
            pos (int, optional): The position in the line to replace the value. Defaults to end (-1).

        Returns:
            str: The modified line.

        Raises:
            RuntimeError: If the `pos` argument is not "start" or "end".
        """
        if index == 0:
            pos_label = "start"
        elif index > 0:
            pos_label = f"{index} index"
        else:
            pos_label = "end"

        # it is unlikely that the sign can change:
        if (old_value > 0 and new_value < 0) or (old_value < 0 and new_value > 0):
            self.logger.debug(
                f"Change of sign detected (old: {old_value}, new: {new_value}). Adjusting {new_value} to {new_value*-1}"
            )
            new_value = new_value * -1

        tmp = lines[index].split(delimiter)
        tmp[pos] = tmp[pos].replace(str(old_value), str(new_value))
        tmp = delimiter.join(tmp)

        old_val_float = float(old_value)
        new_val_str = str(new_value)

        if new_val_str not in lines[index] and tmp == lines[index]:
            self.logger.warning(
                f"{old_value} not found at {pos_label} in line {index}. Trying string extractions"
            )
            trailing_newline = lines[index].endswith("\n")
            tmp = lines[index].rstrip("\n").split(delimiter)
            flag = False
            for i, value in enumerate(tmp):
                try:
                    if float(value.strip()) == old_val_float:
                        tmp[i] = new_val_str
                        if flag:
                            self.logger.warning(
                                f"Multiple occurrences of {old_value} in line {index}: {tmp}"
                            )
                            break
                        if not flag:
                            flag = True
                            self.logger.info(f"Changed {old_value} --> {tmp[i]}")

                except ValueError:
                    continue  # skip non-numeric fields
            tmp = delimiter.join(tmp)
            if trailing_newline:
                tmp += "\n"

        if str(new_value) not in tmp:
            self.logger.error(
                f"Unable to replace {old_value} by {new_value} in line {index}"
            )
        else:
            self.logger.debug(
                f"Changed {lines[index].strip(chr(10))} --> {tmp.strip(chr(10))}"
            )
            lines[index] = tmp

        return lines[index]


def uncomment_line(line: str):
    """
    Removes the '$' symbol at the beginning of each line, if it exists,
    ignoring any leading spaces.

    Args:
        line (str): The line to process.

    Returns:
        str: The uncommented line with the '$' removed if it was at the start.
    """
    stripped_line = line.lstrip()
    if stripped_line.startswith("$"):
        return stripped_line[1:]
    else:
        return line


def comment_line(line: str):
    """
    Adds a '$' symbol at the beginning of the line if it doesn't already exist.

    Args:
        line (str): The line to process.

    Returns:
        str: The line with the '$' added at the start if it wasn't there already.
    """
    # Strip leading spaces and check if the line starts with "$"
    stripped_line = line.lstrip()

    # If the line doesn't start with "$", prepend it
    if not stripped_line.startswith("$"):
        return "$ " + line.lstrip()  # Add "$" at the beginning

    return line


if __name__ == "__main__":

    path = "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/PartType_01_Flat/ASaeule.dat"
    f = FileModifier(path, log_level="DEBUG")
    # modify
    f.set_design_parameters({"Fr": 0.01})
    f.set_design_parameters({"p": 0.1})
    f.set_design_parameters({"D": 0.1})
    # f.set_blank_thickness(0.1)
    f.print()

    # restore
    f.set_design_parameters({"Fr": 0.03})
    f.set_design_parameters({"p": 0.5})
    f.set_design_parameters({"D": 0.2})
    f.set_blank_thickness(0.7)
    f.print()

    path = "/d/gitlab/ensima-code/test_data/ensima-data-main/OpenForm/TCO-Benchmark/PartType_04/Einleger.dat"
    f = FileModifier(path, log_level="DEBUG")
    # modify
    Fr = f.design_parameters["Fr"]["values"][-1]
    p = f.design_parameters["p"]["values"][-1]
    f.set_design_parameters({"Fr": 0.01})
    f.set_design_parameters({"p": 0.1})
    f.print()

    # restore
    f.set_design_parameters({"Fr": Fr})
    f.set_design_parameters({"p": p})
    f.print()
