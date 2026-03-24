#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Lightweight Objective-C exception catcher for use from Swift.
/// Swift cannot catch NSException — this helper bridges that gap.
@interface ObjCExceptionCatcher : NSObject

/// Executes the block. Returns YES if it completed normally,
/// NO if an NSException was thrown (exception is silently caught).
+ (BOOL)tryExecute:(void (NS_NOESCAPE ^)(void))block;

/// Executes the block. If an NSException is thrown, calls the catch block
/// with the exception and returns NO.
+ (BOOL)tryExecute:(void (NS_NOESCAPE ^)(void))block
           onCatch:(void (NS_NOESCAPE ^ _Nullable)(NSException *exception))catchBlock;

@end

NS_ASSUME_NONNULL_END
