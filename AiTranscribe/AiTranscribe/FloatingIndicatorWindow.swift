import AppKit
import ObjectiveC

// MARK: - Global Display Cycle Fix (macOS 26 Tahoe)

/// Fixes a macOS 26 Tahoe bug where NSHostingView triggers re-entrant calls to
/// private `_postWindowNeeds*` methods during AppKit's display cycle, causing
/// NSException → SIGABRT on ANY window containing SwiftUI content.
///
/// The bug: During `displayIfNeeded`, AppKit calls `layoutIfNeeded` which posts
/// `windowDidLayout`. NSHostingView observes this and calls `updateAnimatedWindowSize`
/// or `updateConstraints`, which call `setNeedsDisplay/Layout/UpdateConstraints` on
/// views. These propagate to the window's `_postWindowNeeds*` methods, which throw
/// NSException because they detect a re-entrant update during an active display cycle.
///
/// The fix: Swizzle the three `_postWindowNeeds*` methods to call the original
/// implementation inside an Objective-C @try/@catch block. If the original throws
/// (re-entrant display cycle), the exception is caught and a DEFERRED update is
/// scheduled on the next run loop iteration (outside the display cycle). If the
/// original succeeds (normal operation), behavior is 100% unchanged.
///
/// Evidence from crash logs:
/// - CrashLog 1-3: _postWindowNeedsUpdateConstraints throws (floating indicator)
/// - CrashLog 4: _postWindowNeedsLayout throws (floating indicator)
/// - CrashLog 6: _postWindowNeedsDisplay throws (Settings window)
class DisplayCycleFix {

    /// Call once at app launch (in AppDelegate.applicationDidFinishLaunching).
    static func install() {
        swizzle("_postWindowNeedsDisplay", with: #selector(NSWindow.safe_postWindowNeedsDisplay))
        swizzle("_postWindowNeedsUpdateConstraints", with: #selector(NSWindow.safe_postWindowNeedsUpdateConstraints))
        swizzle("_postWindowNeedsLayout", with: #selector(NSWindow.safe_postWindowNeedsLayout))
    }

    private static func swizzle(_ originalName: String, with swizzledSelector: Selector) {
        let originalSelector = NSSelectorFromString(originalName)
        guard let originalMethod = class_getInstanceMethod(NSWindow.self, originalSelector),
              let swizzledMethod = class_getInstanceMethod(NSWindow.self, swizzledSelector) else {
            print("DisplayCycleFix: could not find method \(originalName) — skipping")
            return
        }
        method_exchangeImplementations(originalMethod, swizzledMethod)
    }
}

// MARK: - NSWindow swizzled methods

extension NSWindow {

    /// Swizzled `_postWindowNeedsDisplay`: calls original, catches NSException if it throws.
    /// On catch: schedules a deferred display on the next run loop pass so the update
    /// isn't lost (fixes frozen timer on M2 Macs where the exception fires every frame).
    @objc func safe_postWindowNeedsDisplay() {
        let success = ObjCExceptionCatcher.tryExecute {
            self.safe_postWindowNeedsDisplay() // After swizzle, calls the original
        }
        if !success {
            // Re-entrant call threw — schedule display outside the display cycle
            DispatchQueue.main.async { [weak self] in
                self?.contentView?.needsDisplay = true
            }
        }
    }

    /// Swizzled `_postWindowNeedsUpdateConstraints`: calls original, catches NSException.
    /// On catch: schedules deferred constraint update.
    @objc func safe_postWindowNeedsUpdateConstraints() {
        let success = ObjCExceptionCatcher.tryExecute {
            self.safe_postWindowNeedsUpdateConstraints() // After swizzle, calls the original
        }
        if !success {
            DispatchQueue.main.async { [weak self] in
                self?.contentView?.needsUpdateConstraints = true
            }
        }
    }

    /// Swizzled `_postWindowNeedsLayout`: calls original, catches NSException.
    /// On catch: schedules deferred layout.
    @objc func safe_postWindowNeedsLayout() {
        let success = ObjCExceptionCatcher.tryExecute {
            self.safe_postWindowNeedsLayout() // After swizzle, calls the original
        }
        if !success {
            DispatchQueue.main.async { [weak self] in
                self?.contentView?.needsLayout = true
            }
        }
    }
}

// MARK: - FloatingIndicatorWindow

/// A custom NSWindow subclass for floating indicator windows (recording indicator,
/// session indicator, snap placeholders). This is now a marker class — the global
/// DisplayCycleFix swizzle handles crash prevention for ALL windows via ObjC
/// try/catch with deferred updates.
class FloatingIndicatorWindow: NSWindow {
}
