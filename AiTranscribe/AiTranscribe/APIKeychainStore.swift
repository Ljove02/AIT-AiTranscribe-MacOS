import Foundation
import Security

enum APIKeychainStore {
    private static let service = "com.aitranscribe.api-keys"

    /// Presence cache so UI checks don't pay SecItemCopyMatching IPC on every
    /// SwiftUI body evaluation. All key writes go through this store, so
    /// mutations keep the cache in sync; external edits via Keychain Access
    /// are picked up on the next real lookup.
    private static let cacheLock = NSLock()
    private static var presenceCache: [String: Bool] = [:]

    private static func cachedPresence(providerId: String) -> Bool? {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return presenceCache[providerId]
    }

    private static func cachePresence(_ present: Bool, providerId: String) {
        cacheLock.lock()
        presenceCache[providerId] = present
        cacheLock.unlock()
    }

    static func saveAPIKey(_ apiKey: String, providerId: String) throws {
        let trimmedKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedKey.isEmpty else {
            try deleteAPIKey(providerId: providerId)
            return
        }

        let data = Data(trimmedKey.utf8)
        let query = baseQuery(providerId: providerId)

        let status = SecItemUpdate(query as CFDictionary, [
            kSecValueData as String: data
        ] as CFDictionary)

        if status == errSecItemNotFound {
            var addQuery = query
            addQuery[kSecValueData as String] = data
            addQuery[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
            let addStatus = SecItemAdd(addQuery as CFDictionary, nil)
            guard addStatus == errSecSuccess else {
                throw KeychainError.unhandledStatus(addStatus)
            }
            cachePresence(true, providerId: providerId)
            return
        }

        guard status == errSecSuccess else {
            throw KeychainError.unhandledStatus(status)
        }
        cachePresence(true, providerId: providerId)
    }

    static func apiKey(providerId: String) throws -> String? {
        var query = baseQuery(providerId: providerId)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)

        if status == errSecItemNotFound {
            cachePresence(false, providerId: providerId)
            return nil
        }

        guard status == errSecSuccess else {
            throw KeychainError.unhandledStatus(status)
        }

        guard let data = item as? Data else {
            cachePresence(false, providerId: providerId)
            return nil
        }

        let key = String(data: data, encoding: .utf8)
        cachePresence(key?.isEmpty == false, providerId: providerId)
        return key
    }

    static func hasAPIKey(providerId: String) -> Bool {
        if let cached = cachedPresence(providerId: providerId) {
            return cached
        }
        return (try? apiKey(providerId: providerId))?.isEmpty == false
    }

    static func deleteAPIKey(providerId: String) throws {
        let status = SecItemDelete(baseQuery(providerId: providerId) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.unhandledStatus(status)
        }
        cachePresence(false, providerId: providerId)
    }

    private static func baseQuery(providerId: String) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: providerId
        ]
    }
}

enum KeychainError: LocalizedError {
    case unhandledStatus(OSStatus)

    var errorDescription: String? {
        switch self {
        case .unhandledStatus(let status):
            return "Keychain operation failed with status \(status)."
        }
    }
}
