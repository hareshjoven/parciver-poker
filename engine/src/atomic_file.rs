//! Publish complete snapshots without truncating the last usable artifact.
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Single-writer snapshot publication. The sibling temporary is synced before
/// rename; readers see either complete generation. Retain the previous generation
/// as `<name>.previous` when requested. On any write/backup/rename error the
/// current destination remains usable. Filesystem/power-loss guarantees still
/// depend on the platform; a killed writer can leave an unused `.tmp` sibling.
pub(crate) fn save(
    path: &Path,
    keep_previous: bool,
    mut write: impl FnMut(&mut BufWriter<File>) -> io::Result<()>,
) -> io::Result<()> {
    save_impl(path, keep_previous, &mut write)
}

fn save_impl(
    path: &Path,
    keep_previous: bool,
    write: &mut dyn FnMut(&mut BufWriter<File>) -> io::Result<()>,
) -> io::Result<()> {
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "snapshot needs a file name"))?;
    let (temp, file) = loop {
        let mut temp_name = name.to_os_string();
        temp_name.push(format!(
            ".{}.{}.tmp",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        let temp_path = path.with_file_name(temp_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => break (TempFile(temp_path), file),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    };
    let mut writer = BufWriter::new(file);
    write(&mut writer)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer); // Windows rename must not depend on our writer handle.

    if keep_previous {
        match File::open(path) {
            Ok(mut current) => {
                let mut previous_name = name.to_os_string();
                previous_name.push(".previous");
                save_impl(&path.with_file_name(previous_name), false, &mut |backup| {
                    io::copy(&mut current, backup)?;
                    Ok(())
                })?;
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
    }
    std::fs::rename(&temp.0, path)?;
    #[cfg(unix)]
    File::open(
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )?
    .sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_write_preserves_current_and_previous_generations() {
        let dir = std::env::temp_dir().join(format!(
            "parciver_atomic_{}_{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("checkpoint.bin");
        let previous = dir.join("checkpoint.bin.previous");
        save(&path, true, |w| w.write_all(b"first")).unwrap();
        save(&path, true, |w| w.write_all(b"second")).unwrap();
        let error = save(&path, true, |w| {
            w.write_all(b"partial third")?;
            Err(io::Error::other("injected write failure"))
        });
        assert!(error.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"second");
        assert_eq!(std::fs::read(&previous).unwrap(), b"first");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(previous).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }
}
