#import "ObjCExceptionCatcher.h"

@implementation ObjCExceptionCatcher

+ (BOOL)tryExecute:(void (NS_NOESCAPE ^)(void))block {
    return [self tryExecute:block onCatch:nil];
}

+ (BOOL)tryExecute:(void (NS_NOESCAPE ^)(void))block
           onCatch:(void (NS_NOESCAPE ^ _Nullable)(NSException *exception))catchBlock {
    @try {
        block();
        return YES;
    }
    @catch (NSException *exception) {
        if (catchBlock) {
            catchBlock(exception);
        }
        return NO;
    }
}

@end
