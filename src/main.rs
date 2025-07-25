use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rustfft::{FftPlanner, num_complex::Complex};
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::error::Error;
use std::f32;
use std::fmt::{Display, Formatter};
use std::io::IsTerminal;
use std::io::Write;
use std::path::Path;
use std::str::FromStr;

type AppError = Box<dyn Error>;

const A2R_A2R1_HEADER: u32 = 0x31523241;
const A2R_A2R2_HEADER: u32 = 0x32523241;
const A2R_A2R3_HEADER: u32 = 0x33523241;

const A2R_INFO_CHUNK: u32 = 0x4f464e49;
const A2R_RWCP_CHUNK: u32 = 0x50435752;
const A2R_SLVD_CHUNK: u32 = 0x44564c53;
const A2R_STRM_CHUNK: u32 = 0x4d525453;
const A2R_DATA_CHUNK: u32 = 0x41544144;
const A2R_META_CHUNK: u32 = 0x4154454d;

const LOOP_POINT_DELTA: usize = 64000;

/// Maximum allowed mismatch ratio when comparing two tracks
const MAX_MISMATCH_RATIO: f64 = 0.001; // 0.1 %
const LABEL: &str = "A2RWOZ";

#[derive(Default)]
struct WozTrackEntry {
    loop_found: bool,
    flux_data: Vec<u8>,
    loop_accuracy: usize,
    capture_type: u8,
    original_flux_data: Vec<u8>,
}

fn parse_creator(s: &str) -> Result<String, String> {
    if s.len() > 32 {
        Err(format!(
            "Creator cannot exceed 32 characters (got {} characters)",
            s.len()
        ))
    } else {
        Ok(format!("{: <width$}", s, width = 32))
    }
}

#[derive(Parser, Debug, Clone)]
#[command(version, about)]
struct Args {
    /// Set the creator (max 32 bytes. If less than 32 bytes, it will be padded with space)
    #[arg(long, default_value = "A2RWOZ", value_parser = parse_creator)]
    creator: String,

    /// Set woz to true to dump tracks in TMAP
    #[arg(long, default_value_t = false)]
    woz: bool,

    /// Dump all tracks in 0.25 tracks increments
    #[arg(long, default_value_t = false)]
    full_tracks: bool,

    /// Duplicate to quarter tracks
    #[arg(long, default_value_t = false)]
    duplicate_quarter_tracks: bool,

    /// Compare tracks
    #[arg(long, default_value_t = false)]
    compare_tracks: bool,

    /// Convert track to flux
    #[arg(long, value_parser = parse_track_ranges)]
    flux: Option<std::vec::Vec<u8>>,

    /// Convert track to tmap
    #[arg(long, value_parser = parse_track_ranges)]
    tmap: Option<std::vec::Vec<u8>>,

    /// Bit Timing
    #[arg(long, default_value_t = 32)]
    bit_timing: u8,

    /// Only return the first loop on each track
    #[arg(long, default_value_t = false)]
    fast_loop: bool,

    /// Enable fallback if loop not found
    #[arg(long, default_value_t = false)]
    enable_fallback: bool,

    /// If loop not found, specified tracks will be added
    #[arg(long, value_parser = parse_track_ranges)]
    tracks: Option<std::vec::Vec<u8>>,

    /// Disable timing
    #[arg(long, default_value_t = false)]
    disable_timing: bool,

    /// Disable xtiming
    #[arg(long, default_value_t = false)]
    disable_xtiming: bool,

    /// Use fft to find loop if loop not found
    #[arg(long)]
    use_fft: bool,

    /// Delete specified track onwards
    #[arg(long, value_parser = parse_track_ranges)]
    delete_tracks: Option<std::vec::Vec<u8>>,

    /// Show information on loop not found on tracks
    #[arg(long)]
    show_failed_loop: bool,

    /// Show tracks that are not solved
    #[arg(long)]
    show_unsolved_tracks: bool,

    /// Enable debug
    #[arg(long)]
    debug: bool,

    #[arg(index = 1, value_name = "input.a2r")]
    input: String,

    #[arg(index = 2, value_name = "output.woz")]
    output: String,
}

#[derive(Default)]
struct Info {
    version: u8,
    creator: String,
    disk_type: u8,
    write_protected: bool,
    synchronized: bool,
    hard_sector_count: u8,
}

impl Display for Info {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        let version = match self.version {
            1 => "1 (A2R1)",
            2 => "2 (A2R2)",
            3 => "3 (A2R3)",
            _ => "Unknown",
        };
        let info = format!("{}INFO{}", green_color(false), reset_color(false));
        writeln!(f, "{info}: {:<20}: {version}", "Version")?;

        let disk_type = match self.disk_type {
            1 => "5.25-inch (140K)",
            2 => "3.5-inch",
            _ => "Unknown",
        };
        writeln!(f, "{info}: {:<20}: {disk_type}", "Disk type")?;
        writeln!(
            f,
            "{info}: {:<20}: {}",
            "Write protected", self.write_protected
        )?;
        writeln!(
            f,
            "{info}: {:<20}: {}",
            "Tracks synchronized", self.synchronized
        )?;
        Ok(())
    }
}

#[derive(PartialEq)]
enum Capture {
    Rwcp,
    Slvd,
    Strm,
    Data,
}

macro_rules! a2_debug {
    ($debug:tt) => {
        if $debug {
            eprintln!()
        }
    };
    ($debug:tt,$($arg:tt)*) => {{
        if $debug {
            eprint!("{LABEL} - ");
            eprintln!($($arg)*);
        }
    }};
}

fn green_color(stderr: bool) -> &'static str {
    if is_terminal(stderr) { "\x1b[32m" } else { "" }
}

fn red_color(stderr: bool) -> &'static str {
    if is_terminal(stderr) { "\x1b[31m" } else { "" }
}

fn reset_color(stderr: bool) -> &'static str {
    if is_terminal(stderr) { "\x1b[0m" } else { "" }
}

fn is_terminal(stderr: bool) -> bool {
    if !stderr {
        std::io::stdout().is_terminal()
    } else {
        std::io::stderr().is_terminal()
    }
}

fn read_a2r_u32(dsk: &[u8], offset: usize) -> u32 {
    dsk[offset] as u32
        + (dsk[offset + 1] as u32) * 256
        + (dsk[offset + 2] as u32) * 65536
        + (dsk[offset + 3] as u32) * 16777216
}

fn read_a2r_big_u32(dsk: &[u8], offset: usize) -> u32 {
    dsk[offset + 3] as u32
        + (dsk[offset + 2] as u32) * 256
        + (dsk[offset + 1] as u32) * 65536
        + (dsk[offset] as u32) * 16777216
}

fn write_woz_u32(dsk: &mut Vec<u8>, value: u32) {
    dsk.push((value & 0xff) as u8);
    dsk.push(((value >> 8) & 0xff) as u8);
    dsk.push(((value >> 16) & 0xff) as u8);
    dsk.push(((value >> 24) & 0xff) as u8);
}

fn write_woz_u16(dsk: &mut Vec<u8>, value: u16) {
    dsk.push((value & 0xff) as u8);
    dsk.push(((value >> 8) & 0xff) as u8);
}

fn process_info(data: &[u8], offset: usize, info: &mut Info, a2r3: bool) {
    if a2r3 {
        info.version = 3;
    } else {
        info.version = 2;
    }
    info.creator = str::from_utf8(&data[offset + 1..offset + 33])
        .unwrap()
        .into();
    info.disk_type = data[offset + 33];
    info.write_protected = data[offset + 34] == 1;
    info.synchronized = data[offset + 35] == 1;

    if a2r3 {
        info.hard_sector_count = data[offset + 36];
    }
}

fn get_label(capture: &Capture) -> &'static str {
    match *capture {
        Capture::Strm => "STRM",
        Capture::Data => "DATA",
        Capture::Rwcp => "RWCP",
        Capture::Slvd => "SLVD",
    }
}

fn process_flux_capture<F>(
    data_input: (&[u8], &mut usize, &mut Args),
    capture: &Capture,
    break_condition: u8,
    speed_offset_calc: impl Fn(usize) -> usize,
    read_data_fn: F,
    hard_sector_count: Option<u8>,
) -> Vec<WozTrackEntry>
where
    F: Fn(&[u8], usize, &Capture, Option<u8>) -> (u8, u8, u32, u32, usize),
{
    let (data, offset, args) = data_input;
    let mut woz_track = Vec::new();
    for _ in 0..160 {
        woz_track.push(WozTrackEntry::default());
    }

    let label = get_label(capture);
    let debug = args.debug;

    let estimated_bit_timing = get_speed(data, speed_offset_calc(*offset));
    a2_debug!(
        debug,
        "{label}: Estimated bit timing : {}",
        estimated_bit_timing
    );

    if args.bit_timing == 0 {
        args.bit_timing = estimated_bit_timing
    }

    let bar = create_progress_bar((data.len() - *offset) as u64);
    let initial = *offset;

    while *offset < data.len() {
        bar.set_position((*offset - initial) as u64);

        let current_byte = data[*offset];
        if current_byte == break_condition {
            break;
        }

        let (location, capture_type, length, loop_point, bytes_read) =
            read_data_fn(data, *offset, capture, hard_sector_count);

        let msg = format!("{}", location as f32 / 4.0);
        bar.set_message(msg);

        *offset += bytes_read;

        process_flux_data(
            data,
            offset,
            length,
            &mut woz_track,
            (location, capture_type, loop_point),
            capture,
            args,
        );
    }
    bar.finish_and_clear();
    woz_track
}

fn process_strm_data(
    data: &[u8],
    offset: &mut usize,
    capture: Capture,
    args: &mut Args,
) -> Vec<WozTrackEntry> {
    let speed_offset_calc = |current_offset: usize| current_offset + 10;
    let read_data_fn = |data: &[u8], current_offset: usize, capture: &Capture, _: Option<u8>| {
        let location = data[current_offset];
        let capture_type = data[current_offset + 1];
        let length = if *capture == Capture::Strm {
            read_a2r_u32(data, current_offset + 2)
        } else {
            read_a2r_big_u32(data, current_offset + 2)
        };
        let loop_point = if *capture == Capture::Strm {
            read_a2r_u32(data, current_offset + 6)
        } else {
            read_a2r_big_u32(data, current_offset + 6)
        };
        (location, capture_type, length, loop_point, 10) // 10 bytes read for this header
    };

    process_flux_capture(
        (data, offset, args),
        &capture,
        0xff,
        speed_offset_calc,
        read_data_fn,
        None,
    )
}

fn process_rwcp_slvd(
    data: &[u8],
    offset: &mut usize,
    args: &mut Args,
    hard_sector_count: u8,
    rwcp: bool,
) -> Vec<WozTrackEntry> {
    let capture_type_enum = if rwcp { Capture::Rwcp } else { Capture::Slvd };

    let speed_offset_calc =
        |current_offset: usize| current_offset + 9 + 4 * data[current_offset + 4] as usize;

    let read_data_fn =
        |data: &[u8], current_offset: usize, _capture: &Capture, hard_sector_count: Option<u8>| {
            let capture_type = data[current_offset + 1];
            let location =
                (data[current_offset + 2] as usize + data[current_offset + 3] as usize * 256) as u8;
            let index_signals_size = data[current_offset + 4] as usize;

            let mut bytes_read = 5; // location, capture_type, location (2 bytes), index_signals_size

            // Skip Mirror Distance Outward and Mirror Distance Inward for SLVD
            if !rwcp {
                bytes_read += 7;
            }

            let mut index_signals = vec![0_usize; index_signals_size];
            if index_signals_size > 0 {
                for item in index_signals.iter_mut().take(index_signals_size) {
                    *item = read_a2r_u32(data, current_offset + bytes_read) as usize;
                    bytes_read += 4;
                }
            }

            let length = read_a2r_u32(data, current_offset + bytes_read);
            bytes_read += 4; // for length

            let loop_point = if index_signals_size == 0 {
                length
            } else if rwcp {
                index_signals[hard_sector_count.unwrap() as usize] as u32
            } else {
                index_signals[0] as u32
            };

            (location, capture_type, length, loop_point, bytes_read)
        };

    *offset += 16;
    process_flux_capture(
        (data, offset, args),
        &capture_type_enum,
        0x58,
        speed_offset_calc,
        read_data_fn,
        Some(hard_sector_count),
    )
}

fn process_flux_data(
    data: &[u8],
    offset: &mut usize,
    length: u32,
    woz_track: &mut [WozTrackEntry],
    flux_info: (u8, u8, u32),
    capture: &Capture,
    args: &mut Args,
) {
    let location = flux_info.0;
    let capture_type = flux_info.1;

    let tracks: Vec<u8> = if let Some(tracks) = &args.tracks {
        tracks.to_vec()
    } else {
        Vec::new()
    };

    if !args.full_tracks && location % 4 != 0 && !tracks.contains(&location) {
        *offset += length as usize;
        return;
    }

    // Only timing and xtiming supported
    if capture_type != 1 && capture_type != 3 {
        *offset += length as usize;
        return;
    }

    // Disable timing if switch disable-timing provided
    if args.disable_timing && capture_type == 1 {
        *offset += length as usize;
        return;
    }

    // Disable xtiming if switch disable-xtiming provided
    if args.disable_xtiming && capture_type == 3 {
        *offset += length as usize;
        return;
    }

    // If accuracy is 100% and if fast loop is enabled, skip the other same locations
    if args.fast_loop && woz_track[location as usize].loop_accuracy == 10000 {
        *offset += length as usize;
        return;
    }

    analyze_flux_data(data, woz_track, flux_info, *offset, length, capture, args);

    *offset += length as usize;
}

fn analyze_flux_data(
    data: &[u8],
    woz_track: &mut [WozTrackEntry],
    flux_info: (u8, u8, u32),
    offset: usize,
    length: u32,
    capture: &Capture,
    args: &mut Args,
) {
    let tracks: Vec<u8> = if let Some(tracks) = &args.tracks {
        tracks.to_vec()
    } else {
        Vec::new()
    };

    let (location, capture_type, loop_point) = flux_info;
    let label = match capture {
        Capture::Strm => "STRM",
        Capture::Data => "DATA",
        Capture::Rwcp => "RWCP",
        Capture::Slvd => "SLVD",
    };
    let capture_type_str = if capture_type == 1 {
        "timing"
    } else {
        "xtiming"
    };
    let debug = args.debug;
    let bit_timing = args.bit_timing;

    a2_debug!(
        debug,
        "{label}: Track: {} Type: {} Capture Len: {} LoopPoint: {}",
        location as f32 / 4.0,
        capture_type_str,
        length,
        loop_point
    );

    a2_debug!(debug, "Bit timing = {bit_timing}");

    let flux_data = &data[offset..offset + length as usize];
    let decompressed = decrunch_stream(flux_data);
    let gap = get_gap_array(&decompressed);
    let cumulative_gap = get_cumulative_gap_array(&gap);
    let normalized_gap = get_normalized_gap_array(&gap, bit_timing);

    if normalized_gap.len() < 100 {
        return;
    }

    //let decompressed = decrunch_stream_flux(&normalized_gap);

    if let Some((start, end, accuracy)) =
        find_loop(&normalized_gap, location, capture_type, loop_point)
    {
        if start < end && start < cumulative_gap.len() && end < cumulative_gap.len() {
            let start = cumulative_gap[start];
            let end = cumulative_gap[end];

            let loop_flux_data = &decompressed[start..end];
            a2_debug!(
                debug,
                "{label}: Track: {} Len: {} End: {} Accuracy: {}%",
                location as f32 / 4.0,
                loop_flux_data.len(),
                end,
                accuracy as f32 / 100.0
            );

            // Update the track if previous accuracy is not 100% or current accuracy is 100%
            // 1. Prefer xtiming over timing capture type
            // 2. If it is same capture-type, prefer higher accuracy
            // 3. If accuracy = 100% is encountered, replaced with a newer one
            let old_woz_track = &woz_track[location as usize];
            if capture_type > old_woz_track.capture_type
                || (capture_type == old_woz_track.capture_type
                    && accuracy >= old_woz_track.loop_accuracy
                    && !loop_flux_data.is_empty())
            {
                let woz_entry = WozTrackEntry {
                    loop_found: true,
                    flux_data: loop_flux_data.to_vec(),
                    loop_accuracy: accuracy,
                    capture_type,
                    original_flux_data: flux_data.to_vec(),
                };
                woz_track[location as usize] = woz_entry;
            }
        }
    } else if args.use_fft {
        let mut data: Vec<_> = decompressed.clone();
        if data.len() < 2 * (loop_point + LOOP_POINT_DELTA as u32) as usize {
            let len = data.len() * 3 / 4;
            for &item in decompressed.iter().take(len) {
                data.push(item)
            }
        }

        if let Ok(end) = find_loop_point_fft(
            &data,
            (loop_point - LOOP_POINT_DELTA as u32) as usize,
            (loop_point + LOOP_POINT_DELTA as u32) as usize,
        ) {
            let mut loop_flux_data = Vec::new();
            for &item in data.iter().take(end) {
                loop_flux_data.push(item);
            }
            let woz_entry = WozTrackEntry {
                loop_found: false,
                flux_data: loop_flux_data.to_vec(),
                loop_accuracy: 0,
                capture_type,
                original_flux_data: flux_data.to_vec(),
            };
            woz_track[location as usize] = woz_entry;
        }
    } else {
        a2_debug!(
            debug,
            "{label}: {}Track: {} Loop not found. Fallback to original data if required{}",
            red_color(true),
            location as f32 / 4.0,
            reset_color(true)
        );
        if (args.enable_fallback || tracks.contains(&location))
            && !woz_track[location as usize].loop_found
        {
            let woz_entry = WozTrackEntry {
                loop_found: false,
                flux_data: (decompressed[0..loop_point as usize]).to_vec(),
                loop_accuracy: 0,
                capture_type,
                original_flux_data: flux_data.to_vec(),
            };
            woz_track[location as usize] = woz_entry;
        }

        if args.show_failed_loop {
            println!(
                "{label}: {}Track: {} ({}) Loop not found{}",
                red_color(false),
                location as f32 / 4.0,
                location,
                reset_color(false)
            );
        }
    }
}

fn find_loop(
    normalized_gap: &[usize],
    pos: u8,
    _capture_type: u8,
    loop_point: u32,
) -> Option<(usize, usize, usize)> {
    const SAMPLE_SIZE: usize = 100;
    const OFFSET_LIMIT: usize = 256;
    const MAX_ALLOWABLE_INDICES: usize = 1000;
    const ACCURACY_MULTIPLIER: usize = 10000;
    const PERFECT_ACCURACY: usize = 10000;
    if pos >= 160 {
        return None;
    }

    if normalized_gap.is_empty() || normalized_gap.len() < SAMPLE_SIZE {
        return None;
    }

    let cumulative_gap = get_cumulative_gap_array(normalized_gap);
    let loop_point = (loop_point as u64 * 1020484 / 1000000) as usize;
    let lower = cumulative_gap
        .binary_search(&(loop_point.saturating_sub(LOOP_POINT_DELTA)))
        .unwrap_or_else(|idx| idx);
    let upper = cumulative_gap
        .binary_search(&(loop_point + LOOP_POINT_DELTA))
        .unwrap_or_else(|idx| idx);

    let mut result = None;
    for index in 1..OFFSET_LIMIT {
        if index + SAMPLE_SIZE > normalized_gap.len() {
            continue;
        }

        let signature = &normalized_gap[index..index + SAMPLE_SIZE];
        let indices: Vec<usize> = normalized_gap
            .windows(SAMPLE_SIZE)
            .enumerate()
            .skip(lower)
            .take_while(|&(i, _)| i < upper)
            .filter(|&(_, window)| window == signature)
            .map(|(i, _)| i)
            .collect();

        if !indices.is_empty() && indices.len() < MAX_ALLOWABLE_INDICES {
            for i in 0..indices.len() {
                if indices[i] > 0 && index < indices[i] {
                    let segment = &normalized_gap[index..indices[i]];

                    if segment.is_empty() {
                        continue;
                    }

                    let compare_data = &normalized_gap[indices[i]..];
                    let compare_len = compare_data.len().min(segment.len());
                    if compare_len == 0 {
                        continue;
                    }
                    let mut accuracy = segment
                        .iter()
                        .zip(compare_data.iter())
                        .filter(|&(a, b)| a == b)
                        .count();
                    accuracy = accuracy * ACCURACY_MULTIPLIER / compare_len;

                    let update_best_match = if let Some((_, _, old_accuracy)) = result {
                        accuracy >= old_accuracy
                    } else {
                        true
                    };

                    if update_best_match {
                        result = Some((index, indices[i], accuracy));
                    }

                    if accuracy == PERFECT_ACCURACY {
                        return result;
                    }
                }
            }
        }

        if result.is_some() {
            break;
        }
    }
    result
}

fn decrunch_stream(flux_record: &[u8]) -> Vec<u8> {
    let total_output_size: usize = flux_record.iter().map(|&b| b as usize).sum();
    let mut v = Vec::with_capacity(total_output_size);
    let mut count: usize = 0;
    for &b in flux_record {
        count += b as usize;
        if count > 0 && b < 255 {
            v.push(1);
            v.extend(std::iter::repeat_n(0, count - 1));
            count = 0
        }
    }
    v
}

fn _decrunch_stream_flux(flux_record: &[usize]) -> Vec<u8> {
    let total_output_size: usize = flux_record.iter().sum();
    let mut v = Vec::with_capacity(total_output_size);
    for &b in flux_record {
        if b > 0 {
            v.push(1);
            v.extend(std::iter::repeat_n(0, b - 1));
        }
    }
    v
}

fn decrunch_stream_woz(flux_len: usize, flux_record: &[usize], bit_timing: u8) -> Vec<u8> {
    let bit_timing = bit_timing as usize;
    let mut v = Vec::with_capacity(flux_len);
    for &flux_total in flux_record.iter() {
        let flux_total = flux_total / bit_timing;
        if flux_total > 1 {
            v.extend(std::iter::repeat_n(0, flux_total - 1));
        }
        v.push(1);
    }
    v
}

fn crunch_stream(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < data.len() {
        i += 1;
        let mut j: usize = 1;
        while i < data.len() && data[i] == 0 {
            i += 1;
            j += 1;
        }
        while j >= 255 {
            result.push(255);
            j -= 255;
        }
        result.push(j as u8);
    }
    result
}

fn crunch_stream_woz(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    let mut mask = 0x80;
    let mut value = 0;
    let mut i = 0;
    while i < data.len() {
        if data[i] > 0 {
            value |= mask
        }
        mask >>= 1;
        if mask == 0 {
            mask = 0x80;
            result.push(value);
            value = 0;
        }
        i += 1;
    }

    if mask != 0x80 {
        result.push(value);
    }

    result
}

fn get_gap_array(data: &[u8]) -> Vec<usize> {
    let mut gap = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        i += 1;
        let mut j = 1;
        while i < data.len() && data[i] == 0 {
            i += 1;
            j += 1;
        }
        gap.push(j)
    }
    gap
}

fn get_cumulative_gap_array(gap: &[usize]) -> Vec<usize> {
    let mut cumulative_sum = 0;
    gap.iter()
        .map(|&item| {
            cumulative_sum += item;
            cumulative_sum
        })
        .collect()
}

fn get_normalized_gap_array(gap: &[usize], bit_timing: u8) -> Vec<usize> {
    gap.iter()
        .map(|&item| normalized_value(item, bit_timing))
        .collect()
}

fn normalized_value(value: usize, bit_timing: u8) -> usize {
    let bit_timing = bit_timing as usize;
    ((value + (bit_timing / 2)) / bit_timing) * bit_timing
}

fn compare_track(track: &[u8], prev_track: &[u8], bit_timing: u8) -> bool {
    let compare_len = track.len().min(prev_track.len());
    let track = &track[0..compare_len];
    let prev_track = &prev_track[0..compare_len];
    let mismatches: usize = track
        .iter()
        .zip(prev_track.iter())
        .map(|(&a, &b)| {
            (normalized_value(a as usize, bit_timing) != normalized_value(b as usize, bit_timing))
                as usize
        })
        .sum();
    (mismatches as f64 / compare_len as f64) < MAX_MISMATCH_RATIO
}

fn _cross_correlation_sameness_ratio_fft(arr1: &[u8], arr2: &[u8]) -> f64 {
    // Handle empty arrays
    if arr1.is_empty() || arr2.is_empty() {
        return 0.0;
    }

    let n1 = arr1.len();
    let n2 = arr2.len();

    // Determine the required padded length for linear cross-correlation
    // This should be n1 + n2 - 1. For FFT efficiency, pad to the next power of 2.
    let min_padded_len = n1 + n2 - 1;
    let fft_size = min_padded_len.next_power_of_two();

    // Convert u8 arrays to Complex<f64> and zero-pad
    let mut input1: Vec<Complex<f64>> = arr1.iter().map(|&x| Complex::new(x as f64, 0.0)).collect();
    input1.resize(fft_size, Complex::new(0.0, 0.0));

    // For cross-correlation, we need to reverse the second array for convolution
    // (x * y)[n] = sum_m x[m] * y[n - m].
    // For cross-correlation (x_corr y)[n] = sum_m x[m] * y[n + m].
    // If h[m] = y[-m], then (x * h)[n] = sum_m x[m] * h[n - m] = sum_m x[m] * y[-(n - m)] = sum_m x[m] * y[m - n].
    // This is equivalent to cross-correlation if the output is reversed and shifted.
    // A simpler way with FFTs is to take the FFT of arr1, FFT of arr2,
    // conjugate FFT(arr2), multiply, then IFFT.
    // Or, for real signals, just reverse arr2 and perform convolution
    let mut input2: Vec<Complex<f64>> = arr2.iter().map(|&x| Complex::new(x as f64, 0.0)).collect();
    input2.resize(fft_size, Complex::new(0.0, 0.0));

    // Create a planner
    let mut planner = FftPlanner::<f64>::new();

    // Create FFT plans
    let fft_forward = planner.plan_fft_forward(fft_size);
    let fft_inverse = planner.plan_fft_inverse(fft_size);

    // Perform forward FFTs
    fft_forward.process(&mut input1);
    fft_forward.process(&mut input2);

    // Point-wise multiplication in frequency domain for cross-correlation
    // Cross-correlation(A, B) <=> IFFT( FFT(A) * conj(FFT(B)) )
    let mut cross_corr_freq: Vec<Complex<f64>> = input1
        .iter()
        .zip(input2.iter())
        .map(|(&a, &b)| {
            a * b.conj() // Multiply FFT(arr1) by conjugate of FFT(arr2)
        })
        .collect();

    // Perform inverse FFT
    fft_inverse.process(&mut cross_corr_freq);

    // Extract the real part of the cross-correlation results
    // And normalize by 1/fft_size as rustfft doesn't normalize by default.
    let cross_corr: Vec<f64> = cross_corr_freq
        .iter()
        .map(|c| c.re / fft_size as f64)
        .collect();

    // Find the absolute maximum value of the cross-correlation
    let mut max_cross_corr_value = 0.0;
    for &val in cross_corr.iter() {
        if val.abs() > max_cross_corr_value {
            max_cross_corr_value = val.abs();
        }
    }

    // Calculate the L2-norm (Euclidean norm) for each input array
    let norm_arr1 = arr1.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_arr2 = arr2.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();

    // Handle division by zero if an array is all zeros
    if norm_arr1 == 0.0 || norm_arr2 == 0.0 {
        return 0.0;
    }

    // Divide the absolute maximum cross-correlation by the product of the norms
    max_cross_corr_value / (norm_arr1 * norm_arr2)
}

fn get_speed(data: &[u8], offset: usize) -> u8 {
    let mut max_index = 0;
    let mut max_value = 0;
    for i in 27..=35 {
        let value = data[offset..]
            .iter()
            .take(8000)
            .map(|&item| (item == i) as usize)
            .sum();
        if value > max_value {
            max_value = value;
            max_index = i;
        }
    }

    if max_index == 0 {
        max_index = 32
    }

    max_index
}

fn main() -> Result<(), AppError> {
    let mut args = Args::parse();

    let now = std::time::Instant::now();
    let dsk = std::fs::read(&args.input)?;

    // Check for WOZ format
    let header = read_a2r_u32(&dsk, 0);

    if header != A2R_A2R3_HEADER && header != A2R_A2R2_HEADER && header != A2R_A2R1_HEADER {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid A2R file",
        )));
    }

    let a2r1 = header == A2R_A2R1_HEADER;
    let a2r3 = header == A2R_A2R3_HEADER;

    let mut dsk_offset = if a2r1 { 16 } else { 8 };
    let mut info = Info::default();
    let mut info_detected = a2r1;
    info.version = 1;

    let data_dsk_offset = dsk_offset;
    while dsk_offset < dsk.len() {
        let chunk_id = read_a2r_u32(&dsk, dsk_offset);
        let chunk_size = if a2r1 {
            read_a2r_big_u32(&dsk, dsk_offset + 4)
        } else {
            read_a2r_u32(&dsk, dsk_offset + 4)
        };
        dsk_offset += 8;
        match chunk_id {
            // INFO
            A2R_INFO_CHUNK => {
                if !a2r1 {
                    process_info(&dsk, dsk_offset, &mut info, a2r3);
                    print!("{info}");
                    if info.disk_type == 1 {
                        info_detected = true;
                    }
                } else {
                    print!("{info}");
                    let metadata = Some(process_meta(&dsk, dsk_offset, chunk_size));
                    if metadata.is_some() {
                        print_meta_information(&metadata, false);
                    }
                }
                dsk_offset += chunk_size as usize;
            }
            A2R_META_CHUNK => {
                let metadata = Some(process_meta(&dsk, dsk_offset, chunk_size));
                if metadata.is_some() {
                    print_meta_information(&metadata, true);
                }
                dsk_offset += chunk_size as usize;
            }
            _ => dsk_offset += chunk_size as usize,
        }
    }

    dsk_offset = data_dsk_offset;
    if info_detected {
        while dsk_offset < dsk.len() {
            let chunk_id = read_a2r_u32(&dsk, dsk_offset);
            let chunk_size = if a2r1 {
                read_a2r_big_u32(&dsk, dsk_offset + 4)
            } else {
                read_a2r_u32(&dsk, dsk_offset + 4)
            };
            dsk_offset += 8;
            match chunk_id {
                // DATA
                A2R_DATA_CHUNK if a2r1 => {
                    let mut woz_tracks =
                        process_strm_data(&dsk, &mut dsk_offset, Capture::Data, &mut args);
                    create_woz_file(&mut woz_tracks, &info, &mut args)?;
                }

                // STRM
                A2R_STRM_CHUNK if info_detected => {
                    let mut woz_tracks =
                        process_strm_data(&dsk, &mut dsk_offset, Capture::Strm, &mut args);
                    create_woz_file(&mut woz_tracks, &info, &mut args)?;
                }

                // RWCP
                A2R_RWCP_CHUNK => {
                    let mut woz_tracks = process_rwcp_slvd(
                        &dsk,
                        &mut dsk_offset,
                        &mut args,
                        info.hard_sector_count,
                        true,
                    );
                    create_woz_file(&mut woz_tracks, &info, &mut args)?;
                }

                // SLVD
                A2R_SLVD_CHUNK => {
                    let mut woz_tracks = process_rwcp_slvd(
                        &dsk,
                        &mut dsk_offset,
                        &mut args,
                        info.hard_sector_count,
                        false,
                    );
                    create_woz_file(&mut woz_tracks, &info, &mut args)?;
                }

                _ => dsk_offset += chunk_size as usize,
            }
        }
    }

    println!();
    if info.disk_type != 1 {
        println!("Only Disk Type 5.25 is supported");
    }
    println!("Elapsed time = {:?}", now.elapsed());

    Ok(())
}

fn process_meta(dsk: &[u8], offset: usize, chunk_size: u32) -> Cow<'_, str> {
    String::from_utf8_lossy(&dsk[offset..offset + chunk_size as usize])
}

fn print_meta_information(meta: &Option<Cow<'_, str>>, metainfo: bool) {
    if let Some(meta) = meta {
        let label = if metainfo { "META" } else { "INFO" };
        for row_item in meta.split('\n') {
            let value: Vec<&str> = row_item.splitn(2, '\t').collect();
            if value.len() > 1 && !value[1].is_empty() {
                println!(
                    "{}{label}{}: {:<20}: {}",
                    green_color(false),
                    reset_color(false),
                    capitalize_first_letter(value[0].trim()),
                    value[1]
                );
            }
        }
    }
}

fn capitalize_first_letter(s: &str) -> String {
    s[0..1].to_uppercase() + &s[1..]
}

fn create_woz_file(
    woz_tracks: &mut [WozTrackEntry],
    woz_info: &Info,
    args: &mut Args,
) -> std::io::Result<()> {
    fn make_chunk(chunk_id: &str, data: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend(chunk_id.as_bytes());
        write_woz_u32(&mut v, data.len() as u32);
        v.extend(data);
        v
    }

    if args.show_unsolved_tracks {
        for (i, item) in woz_tracks.iter().enumerate().take(160) {
            if !woz_tracks[i].flux_data.is_empty() && woz_tracks[i].loop_accuracy != 10000 {
                println!(
                    "{}LOOP{}: Track {:5} ({:3})   : {}% accurate",
                    green_color(false),
                    reset_color(false),
                    i as f32 / 4.0,
                    i,
                    item.loop_accuracy / 100
                )
            }
        }
    }

    let block_size = 512;

    let path = Path::new(&args.output);
    let mut header = Vec::<u8>::new();
    header.extend("WOZ2".as_bytes());
    header.extend(&[0xff, 0xa, 0xd, 0xa]);
    header.extend(&[0, 0, 0, 0]);
    let mut info = vec![0u8; 60];
    info[0] = 3;
    info[1] = 1;
    info[2] = woz_info.write_protected as u8;
    info[3] = woz_info.synchronized as u8;
    info[4] = 0;
    info[5..(5 + 32)].copy_from_slice(args.creator.as_bytes());
    info[37] = 1;
    info[38] = 0;
    info[39] = 32;
    info[40..42].copy_from_slice(&[0x7f, 0]);
    info[42..44].copy_from_slice(&[48, 0]);

    let info_chunk = make_chunk("INFO", &info);

    let mut tmap_map_data = vec![0xff_u8; 160];
    let mut flux_map_data = vec![0xff_u8; 160];
    let mut track_index = 0;
    let mut flux_enabled = false;
    let mut processed_tracks: Vec<u8> = Vec::new();
    let ignore_tracks: Vec<usize> = args
        .delete_tracks
        .as_ref()
        .map(|delete_tracks| delete_tracks.iter().map(|&item| item as usize).collect())
        .unwrap_or_default();

    let bar = create_progress_bar(160);
    for i in 0..160 {
        bar.inc(1);
        let msg = format!("{}", i as f32 / 4.0);
        bar.set_message(msg);

        if ignore_tracks.contains(&i) {
            continue;
        }

        if args.compare_tracks {
            let mut found = 0xff;
            if i < woz_tracks.len() && !woz_tracks[i].flux_data.is_empty() {
                for (index, &item) in processed_tracks.iter().enumerate() {
                    let prev_track = &woz_tracks[item as usize].original_flux_data;
                    if compare_track(
                        &woz_tracks[i].original_flux_data,
                        prev_track,
                        args.bit_timing,
                    ) {
                        found = index as u8;
                        break;
                    }
                }

                if found == 0xff {
                    found = processed_tracks.len() as u8;
                    processed_tracks.push(i as u8);
                } else {
                    woz_tracks[i].flux_data.clear();
                }
            }
            if args.woz {
                tmap_map_data[i] = found;
            } else {
                flux_map_data[i] = found;
                flux_enabled = true;
            }
        } else if args.woz {
            if i < woz_tracks.len() && !woz_tracks[i].flux_data.is_empty() {
                tmap_map_data[i] = track_index;
                if args.duplicate_quarter_tracks && i % 4 == 0 {
                    duplicate_tracks(i, track_index, &mut tmap_map_data);
                }
                track_index += 1;
            }
        } else if i < woz_tracks.len() && !woz_tracks[i].flux_data.is_empty() {
            flux_map_data[i] = track_index;
            if args.duplicate_quarter_tracks && i % 4 == 0 {
                duplicate_tracks(i, track_index, &mut flux_map_data);
            }
            track_index += 1;
            flux_enabled = true;
        }
    }
    bar.finish_and_clear();

    if let Some(flux) = &args.flux {
        flux_enabled = true;

        // Convert tracks to flux
        for &item in flux {
            let track_index = tmap_map_data[item as usize];
            flux_map_data[item as usize] = track_index;
            tmap_map_data[item as usize] = 0xff;
        }
    }

    if let Some(tmap) = &args.tmap {
        // Convert tracks to tmap
        for &item in tmap {
            let track_index = flux_map_data[item as usize];
            tmap_map_data[item as usize] = track_index;
            flux_map_data[item as usize] = 0xff;
        }
    }

    let tmap_chunk = make_chunk("TMAP", &tmap_map_data);
    let flux_chunk = make_chunk("FLUX", &flux_map_data);
    let mut track_start_data = 3;
    let mut trks_data = Vec::<u8>::new();
    let mut fluxs = Vec::<u8>::new();
    let mut largest_tmap_track = 0;
    let mut largest_flux_track = 0;

    for i in 0..160 {
        if i < woz_tracks.len() && !woz_tracks[i].flux_data.is_empty() {
            let flux_data = &woz_tracks[i].flux_data;
            let gap = get_gap_array(flux_data);
            let bit_timing = args.bit_timing;
            let normalized_gap = get_normalized_gap_array(&gap, bit_timing);
            let tmap_data = decrunch_stream_woz(flux_data.len(), &normalized_gap, bit_timing);
            let data = if flux_map_data[i] != 0xff {
                &crunch_stream(flux_data)
            } else {
                &crunch_stream_woz(&tmap_data)
            };

            let track_len = data.len().div_ceil(block_size) * block_size;
            let mut track_data = vec![0u8; track_len];
            let starting_block = track_start_data;
            track_data[0..data.len()].copy_from_slice(data);
            let block_count = track_len / block_size;
            let bit_count = if flux_map_data[i] != 0xff {
                data.len()
            } else {
                tmap_data.len()
            };
            track_start_data += block_count;
            fluxs.extend(&track_data);
            let mut trk_data = Vec::new();
            write_woz_u16(&mut trk_data, starting_block as u16);
            write_woz_u16(&mut trk_data, block_count as u16);
            write_woz_u32(&mut trk_data, bit_count as u32);
            trks_data.extend(&trk_data);

            if flux_map_data[i] == 0xff && block_count > largest_tmap_track {
                largest_tmap_track = block_count
            }

            if flux_map_data[i] != 0xff && block_count > largest_flux_track {
                largest_flux_track = block_count
            }
        }
    }
    trks_data.extend(std::iter::repeat_n(0, 160 * 8 - trks_data.len()));
    trks_data.extend(&fluxs);

    let trks_chunk = make_chunk("TRKS", &trks_data);

    // Final assembly
    let mut contents = Vec::new();
    contents.extend(&header);
    contents.extend(&info_chunk);

    contents.extend(&tmap_chunk);
    contents.extend(&trks_chunk);

    if contents.len() % block_size != 0 {
        let padding = block_size - contents.len() % block_size;
        contents.extend(std::iter::repeat_n(0, padding));
    }

    contents[64] = (largest_tmap_track & 0xff) as u8;
    contents[65] = ((largest_tmap_track >> 8) & 0xff) as u8;

    if flux_enabled {
        let flux_block = contents.len() / 512;
        contents[66] = (flux_block & 0xff) as u8;
        contents[67] = ((flux_block >> 8) & 0xff) as u8;
        contents[68] = (largest_flux_track & 0xff) as u8;
        contents[69] = ((largest_flux_track >> 8) & 0xff) as u8;
        contents.extend(&flux_chunk);
    } else {
        // Set the WOZ version to version 2.0
        contents[20] = 2
    }

    let crc = crc32(0, &contents[header.len()..]);
    contents[8] = (crc & 0xff) as u8;
    contents[9] = ((crc >> 8) & 0xff) as u8;
    contents[10] = ((crc >> 16) & 0xff) as u8;
    contents[11] = ((crc >> 24) & 0xff) as u8;

    let mut file = std::fs::File::create(path)?;
    file.write_all(&contents)?;

    Ok(())
}

fn duplicate_tracks(index: usize, track_index: u8, map: &mut [u8]) {
    if index > 0 {
        map[index - 1] = track_index;
    }
    if index + 1 < 160 {
        map[index + 1] = track_index;
    }

    if index >= 8 && index + 2 < 160 {
        map[index + 2] = track_index;
    }
}

fn generate_crc32_table() -> [u32; 256] {
    let polynomial = 0xedb88320;
    let mut table = [0u32; 256];
    for (i, item) in table.iter_mut().enumerate() {
        let mut crc = i as u32;
        for _ in 0..8 {
            crc = if crc & 1 > 0 {
                (crc >> 1) ^ polynomial
            } else {
                crc >> 1
            };
        }
        *item = crc;
    }
    table
}

fn crc32(value: u32, buf: &[u8]) -> u32 {
    let mut crc = value ^ 0xffffffff;
    let crc32_table = generate_crc32_table();

    for data in buf {
        crc = crc32_table[((crc ^ *data as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    crc ^ 0xffffffff
}

fn create_progress_bar(size: u64) -> ProgressBar {
    let bar = ProgressBar::new(size);
    let progress_style = ProgressStyle::default_bar()
        .template("Processing Track:{msg:>5} {bar:35.green/black} {pos:>7}/{len:7}")
        .unwrap();
    bar.set_style(progress_style.progress_chars("\u{2588}\u{258c} "));
    bar
}

/// Finds the disk’s rotational period (loop point) using FFT-based autocorrelation.
///
/// # Arguments
///
/// * `bits` - An iterable of 0s or 1s, representing the data.
/// *  Its length must be at least `2 * max_expected`.
/// * `min_expected` - The minimum expected loop point (search window in bits).
/// * `max_expected` - The maximum expected loop point (search window in bits).
///
/// # Returns
///
/// A `Result` containing the exact bits per revolution as an `isize` on success,
/// or a `Box<dyn Error>` on failure.
///
/// # Errors
///
/// Returns an error if the input `bits` length is less than `2 * max_expected`.
fn find_loop_point_fft(
    bits: &[u8],
    min_expected: usize,
    max_expected: usize,
) -> Result<usize, Box<dyn Error>> {
    let n = bits.len();

    if n < 2 * max_expected {
        return Err("Need >= 2 revolutions worth of data".into());
    }

    // Map 0/1 to f32 and create a Complex vector for FFT
    let x: Vec<_> = bits.iter().map(|&b| Complex::new(b as f32, 0.0)).collect();

    // Create an FFT planner
    let mut planner = FftPlanner::new();
    let fft_forward = planner.plan_fft_forward(n);
    let fft_inverse = planner.plan_fft_inverse(n);

    // ---- Circular (unbiased) autocorrelation via FFT ------------------------

    // 1. Compute FFT of x
    let mut x_fft = x.clone(); // Clone to keep original x for other potential uses
    fft_forward.process(&mut x_fft);

    // 2. Compute X * X.conj()
    let mut ac_complex: Vec<Complex<f32>> = x_fft.iter().map(|&val| val * val.conj()).collect();

    // 3. Compute IFFT of (x_fft * x_fft.conj())
    fft_inverse.process(&mut ac_complex);

    // Extract the real part for autocorrelation
    let mut ac: Vec<f32> = ac_complex.iter().map(|c| c.re).collect();

    // Unbiased scaling: ac = ac / (n - np.arange(n))
    for (i, item) in ac.iter_mut().enumerate().take(n) {
        let divisor = (n - i) as f32;
        if divisor.abs() > f32::EPSILON {
            // Avoid division by zero
            *item /= divisor;
        } else {
            *item = 0.0; // Or handle as appropriate for your application
        }
    }

    // ---- Search the desired range for the highest peak ----------------------
    let mut max_val = f32::NEG_INFINITY;
    let mut peak_idx: usize = 0;

    // Ensure the slice bounds are within the vector's length
    let start_idx = min_expected;
    let end_idx = (max_expected + 1).min(n); // max_expected + 1 for inclusive end

    if start_idx >= end_idx {
        return Err("Search range is invalid or empty".into());
    }

    for (i, item) in ac.iter().enumerate().take(end_idx).skip(start_idx) {
        if *item > max_val {
            max_val = *item;
            peak_idx = i;
        }
    }

    Ok(peak_idx)
}

/// Custom value parser for the `--tracks` argument
///
/// Parses a comma-separated string of track numbers and ranges (e.g., "1,3-5,8").
/// Returns a sorted `Vec<u8>` of unique track numbers.
fn parse_track_ranges(s: &str) -> Result<Vec<u8>, Box<dyn Error + Sync + Send>> {
    let mut tracks = BTreeSet::new();
    for item in s.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }

        if let Some((start_str, end_str)) = item.split_once('-') {
            let start = u8::from_str(start_str.trim())
                .map_err(|_| format!("Invalid number in range start: '{start_str}'"))?;
            let end = u8::from_str(end_str.trim())
                .map_err(|_| format!("Invalid number in range end: '{end_str}'"))?;

            if start > end {
                return Err(
                    format!("Start of range ({start}) cannot be greater than end ({end})").into(),
                );
            }

            tracks.extend(start..=end);
        } else {
            // Handle single track numbers (e.g., "1", "8")
            let track_num =
                u8::from_str(item).map_err(|_| format!("Invalid track number: '{item}'"))?;
            tracks.insert(track_num);
        }
    }
    Ok(tracks.into_iter().collect())
}
