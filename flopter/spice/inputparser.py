from configparser import *
from collections import defaultdict
import re
import sys
from flopter.core import constants as c


class InputParser(ConfigParser):
    """
    Subclass of ConfigParser, edited to parse SPICE input files which are
    sectioned by dollar signed headings.

    Also changes the default comment_prefixes to:
     - the exclamation mark (!) used in Spice input files
     - the '$end' marker used by Spice to denote the end of a section.

    Supports reading in the commented parameters at the end of SPICE2 input
    files. Can also read in a file upon initialisation, with the optional kwarg
    'input_filename'

    Functions the same way as a config parser in all other ways.
    """
    SECT_TMPL_OVERRIDE = r"\$(?P<header>[^\n]+)"
    SECTCRE = re.compile(SECT_TMPL_OVERRIDE, re.VERBOSE)
    BOOLEAN_STATES = {'1': True, 'yes': True, 'true': True, 'on': True, ".true.": True,
                      '0': False, 'no': False, 'false': False, 'off': False, ".false.": False}
    EXPECTED_DUPES = ('rectangle', 'triangle', 'circle', 'diag_reg', 'specie')
    COMMENT_PARAMS_SECTNAME = 'commented_params'

    def __init__(self, *args, comment_prefixes=('!', '$end'), expected_dupes=EXPECTED_DUPES, input_filename=None,
                 read_comments_fl=True, **kwargs):
        self._duplicate_sects = defaultdict(int)
        for dupe in expected_dupes:
            self._duplicate_sects[dupe] = 0
        ConfigParser.__init__(self, *args, comment_prefixes=comment_prefixes, strict=False, **kwargs)
        self.end_comments = None
        if input_filename:
            file = open(input_filename, 'r')
            self.read_file(file, read_comments=read_comments_fl)
            file.close()

    def _read(self, fp, fpname):
        """
        Override of the _read class. Changes how duplicate section headers are
        treated. Expected duplicates are passed and the first section
        encountered is renamed to itself with an appended 0 (e.g. rectangle0).
        All subsequent duplicates are appended increasing numbers (e.g.
        rectangle1, rectangle2 etc.)


        *** Original Documentation ***

        Parse a sectioned configuration file.

        Each section in a configuration file contains a header, indicated by
        a name in square brackets (`[]'), plus key/value options, indicated by
        `name' and `value' delimited with a specific substring (`=' or `:' by
        default).

        Values can span multiple lines, as long as they are indented deeper
        than the first line of the value. Depending on the parser's mode, blank
        lines may be treated as parts of multiline values or ignored.

        Configuration files may include comments, prefixed by specific
        characters (`#' and `;' by default). Comments may appear on their own
        in an otherwise empty line or may be entered in lines holding values or
        section names.
        """
        elements_added = set()
        cursect = None  # None, or a dictionary
        sectname = None
        optname = None
        lineno = 0
        indent_level = 0
        e = None  # None, or an exception
        for lineno, line in enumerate(fp, start=1):
            comment_start = sys.maxsize
            # strip inline comments
            inline_prefixes = {p: -1 for p in self._inline_comment_prefixes}
            while comment_start == sys.maxsize and inline_prefixes:
                next_prefixes = {}
                for prefix, index in inline_prefixes.items():
                    index = line.find(prefix, index + 1)
                    if index == -1:
                        continue
                    next_prefixes[prefix] = index
                    if index == 0 or (index > 0 and line[index - 1].isspace()):
                        comment_start = min(comment_start, index)
                inline_prefixes = next_prefixes
            # strip full line comments
            for prefix in self._comment_prefixes:
                if line.strip().startswith(prefix):
                    comment_start = 0
                    break
            if comment_start == sys.maxsize:
                comment_start = None
            # Additionally strip commas from the end of values
            value = line[:comment_start].strip().rstrip(',')
            if not value:
                if self._empty_lines_in_values:
                    # add empty line to the value, but only if there was no
                    # comment on the line
                    if (comment_start is None and
                                cursect is not None and
                            optname and
                                cursect[optname] is not None):
                        cursect[optname].append('')  # newlines added at join
                else:
                    # empty line marks end of value
                    indent_level = sys.maxsize
                continue
            # continuation line?
            first_nonspace = self.NONSPACECRE.search(line)
            cur_indent_level = first_nonspace.start() if first_nonspace else 0
            if (cursect is not None and optname and
                        cur_indent_level > indent_level):
                cursect[optname].append(value)
            # a section header or option header?
            else:
                indent_level = cur_indent_level
                # is it a section header?
                mo = self.SECTCRE.match(value)
                if mo:
                    sectname = mo.group('header')
                    # is it expected to be a duplicate section?
                    if sectname in self._duplicate_sects and sectname not in self._sections:
                        self._duplicate_sects[sectname] += 1
                        sectname += str(self._duplicate_sects[sectname]-1)
                    if sectname in self._sections:
                        if self._strict and sectname in elements_added:
                            raise DuplicateSectionError(sectname, fpname,
                                                        lineno)
                        # Altered duplicate section handling
                        # Only works in non-strict mode
                        self._duplicate_sects[sectname] += 1
                        while sectname in self._sections:
                            sectname = sectname + str(self._duplicate_sects[sectname])
                        cursect = self._dict()
                        self._sections[sectname] = cursect
                        self._proxies[sectname] = SectionProxy(self, sectname)
                        elements_added.add(sectname)
                    elif sectname == self.default_section:
                        cursect = self._defaults
                    else:
                        cursect = self._dict()
                        self._sections[sectname] = cursect
                        self._proxies[sectname] = SectionProxy(self, sectname)
                        elements_added.add(sectname)
                    # So sections can't start with a continuation line
                    optname = None
                # no section header in the file?
                elif cursect is None:
                    raise MissingSectionHeaderError(fpname, lineno, line)
                # an option line?
                else:
                    mo = self._optcre.match(value)
                    if mo:
                        optname, vi, optval = mo.group('option', 'vi', 'value')
                        if not optname:
                            e = self._handle_error(e, fpname, lineno, line)
                        optname = self.optionxform(optname.rstrip())
                        if (self._strict and
                                    (sectname, optname) in elements_added):
                            raise DuplicateOptionError(sectname, optname,
                                                       fpname, lineno)
                        elements_added.add((sectname, optname))
                        # This check is fine because the OPTCRE cannot
                        # match if it would set optval to None
                        if optval is not None:
                            optval = optval.strip()
                            cursect[optname] = [optval]
                        else:
                            # valueless option handling
                            cursect[optname] = None
                    else:
                        # a non-fatal parsing error occurred. set up the
                        # exception but keep going. the exception will be
                        # raised at the end of the file and will contain a
                        # list of all bogus lines
                        e = self._handle_error(e, fpname, lineno, line)
        # if any parsing errors occurred, raise an exception
        if e:
            raise e
        self._join_multiline_values()

    def write(self, fp, space_around_delimiters=True):
        """Write an .ini-format representation of the configuration state.

        If `space_around_delimiters' is True (the default), delimiters
        between keys and values are surrounded by spaces.
        """
        super().write(fp, space_around_delimiters=space_around_delimiters)

    def _write_section(self, fp, section_name, section_items, delimiter):
        """Write a single section to the specified `fp'."""
        # Treat comment section specially: write out either the saved raw
        # comments or write out the parameters with the comment prefix prepended
        if section_name == self.COMMENT_PARAMS_SECTNAME:
            if self.end_comments is not None:
                for comment in self.end_comments:
                    if len(comment) > 0 and comment[0] == self._comment_prefixes[0]:
                        fp.write(f'{comment} \n')
            else:
                fp.write('! Plasma Parameters:')
                for key, value in section_items:
                    value = self._interpolation.before_write(self, section_name, key, value)
                    if value is not None or not self._allow_no_value:
                        value = delimiter + str(value).replace('\n', '\n\t')
                    else:
                        value = ""
                    fp.write("! {}{}\n".format(key, value))
            return

        # Remove trailing numbers appended for getting duplicate section names into a configparser.
        if section_name[:-1] in self.EXPECTED_DUPES:
            section_name = section_name[:-1]

        # Write section header and parameters
        fp.write("${}\n".format(section_name))
        for key, value in section_items:
            value = self._interpolation.before_write(self, section_name, key,
                                                     value)
            if value is not None or not self._allow_no_value:
                value = delimiter + str(value).replace('\n', '\n\t')
            else:
                value = ""
            fp.write("\t{}{}\n".format(key, value))
        fp.write("$end\n"
                 "\n")

    def read_file(self, f, source=None, read_comments=False):
        super().read_file(f, source)
        if read_comments:
            # get last 10 lines of file
            f.seek(0)
            comments = f.read().split('\n')[-10:]
            self.end_comments = comments
            params = self.read_commented_params(comments)
            self._sections[self.COMMENT_PARAMS_SECTNAME] = params
            self._proxies[self.COMMENT_PARAMS_SECTNAME] = SectionProxy(self, self.COMMENT_PARAMS_SECTNAME)

    def has_commented_params(self):
        """
        Method to check if parser has read in commented parameters by checking
        if the 'commented params' section exists

        :return: Boolean stating if parameters have been read successfully

        """
        return self.COMMENT_PARAMS_SECTNAME in self._sections

    def get_commented_params(self):
        """
        Returns commented parameters if they have been read into the object,
        throws exception if not.

        :return: Dictionary containing commented parameters

        """
        if self.COMMENT_PARAMS_SECTNAME in self._sections:
            return self._sections[self.COMMENT_PARAMS_SECTNAME]
        else:
            raise NoSectionError(self.COMMENT_PARAMS_SECTNAME)

    def has_scanning_params(self):
        """
        Method for checking if config file is being used to initiate a parameter
        scan. If any parameter is specified with a python list (i.e.
        [scan_param_1, scan_param_2, ...]) it is counted as a scanning
        parameter.

        :return:    True if there are any parameters specified as scanning
                    parameters, False otherwise.

        """
        return len(self.get_scanning_params()) > 0

    def get_scanning_params(self):
        """
        Method for returning information about the parameters specified to be
        scanned. Returns a list of dictionaries with four entries for each
        scanning parameter, of the form:
        {
            'section':      [string] name of section where parameter is
            'parameter':    [string] name of parameter being scanned
            'values':       [list] parameter values to be scanned
            'length':       [int] length of the parameter scan, i.e. how many
                            values in the list of parameters
        }

        :return:    list of dicts containing the above data for each parameter

        """
        scan_params = []
        for section, parameters in self._sections.items():
            for param, value in parameters.items():
                # Convert commented params to strings to avoid an error
                if section == self.COMMENT_PARAMS_SECTNAME:
                    value = str(value)
                if '[' in value and ']' in value:
                    value = value.strip('[]').split(', ')
                    scan_params.append({'section': section, 'parameter': param, 'values': value, 'length': len(value)})
        return scan_params

    def read_commented_params(self, comments):
        """
        Reads out the parameters from the end of a SPICE2 input file if they are
        embedded within comments.

        :param comments:    List of strings containing the commented parameters
                            (in form '! T_e = 5.2e2 eV')

        :return:            Dictionary containing commented parameters

        """
        params = {}
        for comment in comments:
            comment_parts = comment.split()
            if (len(comment_parts) < 4
               or comment_parts[0] not in self._comment_prefixes
               or comment_parts[2] != '='):
                continue
            params[comment_parts[1]] = float(comment_parts[3])
        return params

    def get_diag_index(self, diag_name):
        """
        Find index of diagnostic within an input file
        """
        for i in range(len([section for section in self.sections() if c.INF_SEC_DIAG in section]) - 1):
            diag_section_name = c.INF_SEC_DIAG + str(i)
            if self.get(diag_section_name, c.INF_DIAG_NAME).strip('\\\'') == diag_name:
                return i
        print('No diagnostic region found matching "{}"'.format(diag_name))
        return None

    def get_hist_index(self, hist_name):
        """
        Find the index of a diagnostic which is a histogram within an input file
        """
        count = -1
        # Iterate through all diagnostic sections
        for i in range(len([section for section in self.sections() if c.INF_SEC_DIAG in section]) - 1):
            diag_section_name = c.INF_SEC_DIAG + str(i)
            # Checks if the diagnostic is a histogram or not
            if self.getint(diag_section_name, c.INF_DIAG_PROPERTY) == 3:
                count += 1
                if self.get(diag_section_name, c.INF_DIAG_NAME).strip('\\\'') == hist_name:
                    return count
        print('No diagnostic region making histograms found matching "{}"'.format(hist_name))
        return None

    def get_hist_diag_regions(self, species=2):
        """
        Find the simulation coordinates of each diagnostic which is a histogram
        and return them as a dictionary of lists.

        :return:    [list] Dictionary of lists containing the coordinates of
                    the diagnostic region. In the format {diag_name: [z_low,
                    z_high, y_low, y_high], ... }
        """
        diag_regions = {}
        species_diag_name = 'i' if species == 1 else 'e' if species == 2 else '' + c.DIAG_DIST_FUNCTION_HIST

        # Iterate through all diagnostic sections
        for i in range(len([section for section in self.sections() if c.INF_SEC_DIAG in section]) - 1):
            diag_section_name = c.INF_SEC_DIAG + str(i)
            # Checks if the diagnostic is a histogram or not
            if self.getint(diag_section_name, c.INF_DIAG_PROPERTY) == 3 \
                    and species_diag_name in self.get(diag_section_name, c.INF_DIAG_NAME).strip('\\\''):
                diag_name = self.get(diag_section_name, c.INF_DIAG_NAME).strip('\\\'')
                y_low = self.getint(diag_section_name, c.INF2_DIAG_YLOW)
                y_high = self.getint(diag_section_name, c.INF2_DIAG_YHIGH)
                z_low = self.getint(diag_section_name, c.INF2_DIAG_ZLOW)
                z_high = self.getint(diag_section_name, c.INF2_DIAG_ZHIGH)
                diag_regions[diag_name] = [z_low, z_high, y_low, y_high]

        return diag_regions

    def get_probe_obj_indices(self):
        """
        Returns the index (or indices if the probe is a compound shape) of the
        object within the simulation which is acting as a probe i.e. has
        'param1' set to 3. The indices can be used to reference the correct
        array in objectscurrent, for example.

        :return: [int]  Indices of probes in simulation object array, returned
                        as a list.

        """
        num_blocks_section = self[c.INF_SEC_SHAPES]
        n = 0
        probes = []
        for shape in num_blocks_section:
            n_shape = self.getint(c.INF_SEC_SHAPES, shape)
            n += n_shape
            if n_shape > 0:
                # shave off the trailing s from the label
                shape_name = shape[:-1]
                for i in range(n_shape):
                    section = self[shape_name + str(i)]
                    if int(section[c.INF_SWEEP_PARAM]) == c.PROBE_PARAMETER:
                        probes.append((n - n_shape) + i)
        if len(probes) > 0:
            return probes
        else:
            raise ValueError('Could not find a shape set to sweep voltage')

    def get_wall_obj_indices(self):
        """
        Returns the index (or indices if using a compound shape) of the object
        within the simulation which is acting set to have a floating potential
        i.e. has 'param1' set to 2. The indices can be used to reference the
        correct array in objectscurrent, for example.

        :return: [int]  Indices of floating potential walls in simulation object
                        array, returned as a list.

        """
        num_blocks_section = self[c.INF_SEC_SHAPES]
        n = 0
        probes = []
        for shape in num_blocks_section:
            n_shape = self.getint(c.INF_SEC_SHAPES, shape)
            n += n_shape
            if n_shape > 0:
                # shave off the trailing s from the label
                shape_name = shape[:-1]
                for i in range(n_shape):
                    section = self[shape_name + str(i)]
                    if int(section[c.INF_SWEEP_PARAM]) == c.WALL_PARAMETER:
                        probes.append((n - n_shape) + i)
        if len(probes) > 0:
            return probes
        else:
            raise ValueError('Could not find a shape set to sweep voltage')

    def has_sufficient_sweep(self, percentage, voltage_threshold=2):
        """
        Boolean assessment of whether, given a percentage of progress through
        the simulation (taken from the logfile or otherwise), a sufficient
        portion of the sweep has been executed for the simulation to be analysed

        :param percentage:          fraction of completion of a simulation
        :param voltage_threshold:   Absolute value of voltage required to have
                                    been reached to warrant analysis
        :return:                    Returns True if sufficient percentage has
                                    been simulated
        """
        return percentage >= self.get_threshold_fraction(voltage_threshold=voltage_threshold)

    def get_threshold_fraction(self, voltage_threshold=2):
        """
        Returns the fraction of total simulation time required (given a passed
        voltage_threshold) for the simulation to be sufficiently far through as
        to be analysed for ion saturation and temperature.

        :param voltage_threshold:   Absolute value of voltage required to have
                                    been reached to warrant analysis
        """
        if voltage_threshold < c.SWEEP_LOWER or voltage_threshold > c.SWEEP_UPPER:
            raise ValueError('Passed voltage voltage_threshold is outside of sweep range')

        threshold_fraction = (voltage_threshold - c.SWEEP_LOWER) / (c.SWEEP_UPPER - c.SWEEP_LOWER)

        t_a = self.getfloat(c.INF_SEC_GEOMETRY, c.INF_TIME_AV)
        t_p = self.getfloat(c.INF_SEC_GEOMETRY, c.INF_TIME_END)

        return (t_a + (threshold_fraction * (t_p - t_a))) / t_p

    def get_scaling_values(self, len_diag, len_builtin):
        """
        Calculates the scaling values (n' and r) which are needed to extend the
        diagnostic outputs to the right length and downsample them for
        homogenisation of SPICE IV sweeps

        :param len_diag:    length of raw diagnostic output array   (n)
        :param len_builtin: length of builtin output array          (M)
        :return n_leading:  size of array to prepend onto the diagnostic array
        :return ratio:      ratio of extended diagnostic output array to builtin
                            output array (e.g. objectscurrent):

        """
        t_c = self.getfloat(c.INF_SEC_GEOMETRY, c.INF_TIME_SWEEP)
        t_p = self.getfloat(c.INF_SEC_GEOMETRY, c.INF_TIME_END)
        # t_c as a fraction of whole time
        t = t_c/t_p

        n_leading = t * len_diag / (1 - t)
        ratio = len_diag/(len_builtin*(1-t))
        return int(n_leading), int(ratio)

    def get_sweep_length(self, len_builtin, raw_voltage):
        t_a = self.getfloat(c.INF_SEC_GEOMETRY, c.INF_TIME_AV)
        t_p = self.getfloat(c.INF_SEC_GEOMETRY, c.INF_TIME_END)
        # t_a as a fraction of whole time
        t = t_a / t_p

        sweep_length = int(t * len_builtin)

        initial_v = raw_voltage[0]
        if not self._is_within_bounds(initial_v, c.SWEEP_LOWER):
            corr_sweep_length = sweep_length
            while raw_voltage[corr_sweep_length] == initial_v and corr_sweep_length < len(raw_voltage) - 1:
                corr_sweep_length += 1
            sweep_length = corr_sweep_length
        return sweep_length

    def get_probe_geometry(self):
        # TODO: (25/10/2017) Write function to retrieve probe geometry from parser. Probably requires the definition
        # TODO: of a probe-geometry class first.
        # TODO: (22/06/2018) Definition of probe geometery completed, this can now be implemented.
        pass

    @staticmethod
    def _is_within_bounds(value, comparison):
        return (value > comparison - 0.01) and (value < comparison + 0.01)
